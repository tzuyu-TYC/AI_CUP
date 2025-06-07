from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import joblib
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectFromModel

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft

device = torch.device("cpu")



def FFT(xreal, ximag):
    n = 2
    while(n*2 <= len(xreal)):
        n *= 2

    p = int(math.log(n, 2))

    for i in range(0, n):
        a = i
        b = 0
        for j in range(0, p):
            b = int(b*2 + a%2)
            a = a/2
        if(b > i):
            xreal[i], xreal[b] = xreal[b], xreal[i]
            ximag[i], ximag[b] = ximag[b], ximag[i]

    wreal = []
    wimag = []

    arg = float(-2 * math.pi / n)
    treal = float(math.cos(arg))
    timag = float(math.sin(arg))

    wreal.append(float(1.0))
    wimag.append(float(0.0))

    for j in range(1, int(n/2)):
        wreal.append(wreal[-1] * treal - wimag[-1] * timag)
        wimag.append(wreal[-1] * timag + wimag[-1] * treal)

    m = 2
    while(m < n + 1):
        for k in range(0, n, m):
            for j in range(0, int(m/2), 1):
                index1 = k + j
                index2 = int(index1 + m / 2)
                t = int(n * j / m)
                treal = wreal[t] * xreal[index2] - wimag[t] * ximag[index2]
                timag = wreal[t] * ximag[index2] + wimag[t] * xreal[index2]
                ureal = xreal[index1]
                uimag = ximag[index1]
                xreal[index1] = ureal + treal
                ximag[index1] = uimag + timag
                xreal[index2] = ureal - treal
                ximag[index2] = uimag - timag
        m *= 2

    return n, xreal, ximag

def FFT_data(input_data, swinging_times):
    txtlength = swinging_times[-1] - swinging_times[0]
    a_mean = [0] * txtlength
    g_mean = [0] * txtlength

    for num in range(len(swinging_times)-1):
        a = []
        g = []
        for swing in range(swinging_times[num], swinging_times[num+1]):
            a.append(math.sqrt(math.pow((input_data[swing][0] + input_data[swing][1] + input_data[swing][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[swing][3] + input_data[swing][4] + input_data[swing][5]), 2)))
            #a_val = math.sqrt(input_data[swing][0]**2 + input_data[swing][1]**2 + input_data[swing][2]**2) #修改版本，可忽略
            #g_val = math.sqrt(input_data[swing][3]**2 + input_data[swing][4]**2 + input_data[swing][5]**2)
            #a.append(a_val)
            #g.append(g_val)

        a_mean[num] = (sum(a) / len(a))
        g_mean[num] = (sum(a) / len(a))
        #a_mean[num] = sum(a) / len(a) if len(a) > 0 else 0 #修改版本，可忽略
        #g_mean[num] = sum(g) / len(g) if len(g) > 0 else 0

    return a_mean, g_mean

def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer):
    allsum = []
    mean = []
    var = []
    rms = []
    XYZmean_a = 0
    a = []
    g = []
    a_s1 = 0
    a_s2 = 0
    g_s1 = 0
    g_s2 = 0
    a_k1 = 0
    a_k2 = 0
    g_k1 = 0
    g_k2 = 0
    
    for i in range(len(input_data)):
        if i==0:
            allsum = input_data[i]
            a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
            continue
        
        a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
        g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
       
        allsum = [allsum[feature_index] + input_data[i][feature_index] for feature_index in range(len(input_data[i]))]
        
    mean = [allsum[feature_index] / len(input_data) for feature_index in range(len(input_data[i]))]
    
    for i in range(len(input_data)):
        if i==0:
            var = input_data[i]
            rms = input_data[i]
            continue

        var = [var[feature_index] + math.pow((input_data[i][feature_index] - mean[feature_index]), 2) for feature_index in range(len(input_data[i]))]
        rms = [rms[feature_index] + math.pow(input_data[i][feature_index], 2) for feature_index in range(len(input_data[i]))]
        
    #var = [math.sqrt((var[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    #rms = [math.sqrt((rms[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    var = [math.sqrt(max(var[feature_index] / len(input_data), 0)) for feature_index in range(len(input_data[i]))] # 修正成這樣後才能修好3211
    rms = [math.sqrt(max(rms[feature_index] / len(input_data), 0)) for feature_index in range(len(input_data[i]))] 
    
    a_max = [max(a)]
    a_min = [min(a)]
    a_mean = [sum(a) / len(a)]
    g_max = [max(g)]
    g_min = [min(g)]
    g_mean = [sum(g) / len(g)]
    
    a_var = math.sqrt(math.pow((var[0] + var[1] + var[2]), 2))
    
    for i in range(len(input_data)):
        a_s1 = a_s1 + math.pow((a[i] - a_mean[0]), 4)
        a_s2 = a_s2 + math.pow((a[i] - a_mean[0]), 2)
        g_s1 = g_s1 + math.pow((g[i] - g_mean[0]), 4)
        g_s2 = g_s2 + math.pow((g[i] - g_mean[0]), 2)
        a_k1 = a_k1 + math.pow((a[i] - a_mean[0]), 3)
        g_k1 = g_k1 + math.pow((g[i] - g_mean[0]), 3)
    
    a_s1 = a_s1 / len(input_data)
    a_s2 = a_s2 / len(input_data)
    g_s1 = g_s1 / len(input_data)
    g_s2 = g_s2 / len(input_data)
    a_k2 = math.pow(a_s2, 1.5)
    g_k2 = math.pow(g_s2, 1.5)
    a_s2 = a_s2 * a_s2
    g_s2 = g_s2 * g_s2
    
    #a_kurtosis = [a_s1 / a_s2] #修改前版本
    #g_kurtosis = [g_s1 / g_s2]
    #a_skewness = [a_k1 / a_k2]
    #g_skewness = [g_k1 / g_k2]
    a_kurtosis = [a_s1 / a_s2 if a_s2 != 0 else 0] #修改後版本
    g_kurtosis = [g_s1 / g_s2 if g_s2 != 0 else 0] 
    a_skewness = [a_k1 / a_k2 if a_k2 != 0 else 0]
    g_skewness = [g_k1 / g_k2 if g_k2 != 0 else 0] 
    
    a_fft_mean = 0
    g_fft_mean = 0
    cut = int(n_fft / swinging_times)
    a_psd = []
    g_psd = []
    entropy_a = []
    entropy_g = []
    e1 = []
    e3 = []
    e2 = 0
    e4 = 0
    
    for i in range(cut * swinging_now, cut * (swinging_now + 1)):
        a_fft_mean += a_fft[i]
        g_fft_mean += g_fft[i]
        a_psd.append(math.pow(a_fft[i], 2) + math.pow(a_fft_imag[i], 2))
        g_psd.append(math.pow(g_fft[i], 2) + math.pow(g_fft_imag[i], 2))
        e1.append(math.pow(a_psd[-1], 0.5))
        e3.append(math.pow(g_psd[-1], 0.5))
        
    a_fft_mean = a_fft_mean / cut
    g_fft_mean = g_fft_mean / cut
    
    a_psd_mean = sum(a_psd) / len(a_psd)
    g_psd_mean = sum(g_psd) / len(g_psd)
    
    for i in range(cut):
        e2 += math.pow(a_psd[i], 0.5)
        e4 += math.pow(g_psd[i], 0.5)
    
    for i in range(cut):
        #entropy_a.append((e1[i] / e2) * math.log(e1[i] / e2)) #修改前版本
        #entropy_g.append((e3[i] / e4) * math.log(e3[i] / e4))
        if e1[i] > 0 and e2 > 0: #修改後版本
            entropy_a.append((e1[i] / e2) * math.log(e1[i] / e2))
        else:
            entropy_a.append(0)
        if e3[i] > 0 and e4 > 0:
            entropy_g.append((e3[i] / e4) * math.log(e3[i] / e4))
        else:
            entropy_g.append(0) # 修改後版本

    
    a_entropy_mean = sum(entropy_a) / len(entropy_a)
    g_entropy_mean = sum(entropy_g) / len(entropy_g)       
        
    
    output = mean + var + rms + a_max + a_mean + a_min + g_max + g_mean + g_min + [a_fft_mean] + [g_fft_mean] + [a_psd_mean] + [g_psd_mean] + a_kurtosis + g_kurtosis + a_skewness + g_skewness + [a_entropy_mean] + [g_entropy_mean]
    writer.writerow(output)

def data_generate():
    datapath = './train_data'
    tar_dir = 'tabular_data_train'
    os.makedirs(tar_dir, exist_ok=True)
    pathlist_txt = Path(datapath).glob('**/*.txt')

    for file in pathlist_txt:
        f = open(file)
        All_data = []
        count = 0
        for line in f.readlines():
            if line == '\n' or count == 0:
                count += 1
                continue
            num = line.split(' ')
            if len(num) > 5:
                tmp_list = []
                for i in range(6):
                    tmp_list.append(int(num[i]))
                All_data.append(tmp_list)
        f.close()

        swing_index = np.linspace(0, len(All_data), 28, dtype=int)
        headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt', 'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy']

        with open('./{}/{}.csv'.format(tar_dir, Path(file).stem), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headerList)
            try:
                a_fft, g_fft = FFT_data(All_data, swing_index)
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                for i in range(len(swing_index)):
                    if i == 0:
                        continue
                    feature(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)
            except:
                print(Path(file).stem)
                continue

def test_data_generate():
    datapath = './test_data'
    tar_dir = 'tabular_data_test'
    os.makedirs(tar_dir, exist_ok=True)
    pathlist_txt = Path(datapath).glob('**/*.txt')

    for file in pathlist_txt:
        f = open(file)
        All_data = []
        count = 0
        for line in f.readlines():
            if line == '\n' or count == 0:
                count += 1
                continue
            num = line.split(' ')
            if len(num) > 5:
                tmp_list = []
                for i in range(6):
                    tmp_list.append(int(num[i]))
                All_data.append(tmp_list)
        f.close()

        # 補齊長度不足的資料（至少要 28 筆，才能切成 27 段）
        min_len = 28
        if len(All_data) < min_len:
            print(f"補齊 {file}，原始長度為 {len(All_data)}，將補到 {min_len}")
            last_row = All_data[-1] if All_data else [0] * 6
            while len(All_data) < min_len:
                All_data.append(last_row)
        
        swing_index = np.linspace(0, len(All_data), 28, dtype=int)
        headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt', 'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy']

        with open('./{}/{}.csv'.format(tar_dir, Path(file).stem), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headerList)
            try:
                a_fft, g_fft = FFT_data(All_data, swing_index)
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                for i in range(1, len(swing_index)):
                    feature(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1,
                    n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)
            except Exception as e:
                print(f"檔案 {Path(file).stem} 特徵提取失敗，錯誤原因：{e}")
                continue
            #try: #修改前版本
                #a_fft, g_fft = FFT_data(All_data, swing_index) 
                #a_fft_imag = [0] * len(a_fft)
                #g_fft_imag = [0] * len(g_fft)
                #n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                #n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                #for i in range(len(swing_index)):
                    #if i == 0:
                        #continue
                    #feature(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)
            #except:
                #print(Path(file).stem)
                #continue


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, cnn_channels, lstm_hidden_dim, transformer_dim, output_dim,
                 num_heads=8, num_layers=2, kernel_size=3, dropout_rate=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden_dim * 2,  # BiLSTM 輸出維度
            nhead=num_heads,
            dim_feedforward=transformer_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_dim, output_dim)
        )
        #self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)       # [B, T, F] → [B, F, T]
        x = self.cnn(x)             # [B, C, T]
        x = x.transpose(1, 2)       # [B, T, C]
        lstm_out, _ = self.lstm(x)  # [B, T, 2*H]
        x = self.transformer(lstm_out)  # 加入 attention
        out = x[:, -1, :]           # 取最後 timestep 特徵
        return self.fc(out)


def train_bilstm(X_train, y_train, input_dim, output_dim, cnn_channels=32, lstm_hidden_dim=64, transformer_dim=128, num_heads = 8, num_epochs=120):
    model = BiLSTMClassifier(input_dim=input_dim, cnn_channels=cnn_channels, lstm_hidden_dim=lstm_hidden_dim, transformer_dim=transformer_dim,
    output_dim=output_dim, num_heads=num_heads, num_layers=2, dropout_rate=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return model

def predict_bilstm_proba(model, X):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs

def encode_binary_column(series, positive_value):
    return (series == positive_value).astype(int)

def main(): #train and valid

    group_size = 27
    info = pd.read_csv('train_info.csv')
    unique_players = info['player_id'].unique()
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)
    if not os.path.exists('./tabular_data_train'):
        data_generate()
    datapath = './tabular_data_train'
    datalist = list(Path(datapath).glob('**/*.csv'))
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']

    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    x_test = pd.DataFrame()
    y_test = pd.DataFrame(columns=target_mask)
    id_test = []

    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file)
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data), ignore_index=True)
        if player_id in train_players:
            x_train = pd.concat([x_train, data], ignore_index=True)
            y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        elif player_id in test_players:
            x_test = pd.concat([x_test, data], ignore_index=True)
            y_test = pd.concat([y_test, target_repeated], ignore_index=True)
            id_test.append(unique_id)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, 'scaler.pkl')

    y_train_gender = encode_binary_column(y_train['gender'], 1)
    y_test_gender = encode_binary_column(y_test['gender'], 1)

    y_train_hold = encode_binary_column(y_train['hold racket handed'], 1)
    y_test_hold = encode_binary_column(y_test['hold racket handed'], 1)
    
    input_dim = x_train.shape[1]
    X_train_seq = X_train_scaled.reshape(-1, group_size, input_dim)
    X_test_seq = X_test_scaled.reshape(-1, group_size, input_dim)
    
    y_train_years = y_train['play years'].astype(int).values[::group_size]
    y_test_years = y_test['play years'].astype(int).values[::group_size]
    y_train_level = y_train['level'].astype(int).values[::group_size]
    y_test_level = y_test['level'].astype(int).values[::group_size]

    # Shift class labels to start from 0
    y_train_years -= y_train_years.min()
    y_test_years -= y_test_years.min()
    y_train_level -= y_train_level.min()
    y_test_level -= y_test_level.min()
    total_auc = 0
    auc_count = 0

    def model_binary(X_train, y_train, X_test, y_test, label):
        #clf = RandomForestClassifier(random_state=42)
        clf = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=0, depth=2, l2_leaf_reg=5, random_state=42)
        
        nonlocal total_auc, auc_count 
        clf.fit(X_train, y_train)
        joblib.dump(clf, f"{label}_model.pkl")
        probs = clf.predict_proba(X_test)
        y_pred = [p[1] for p in probs]
        y_true = [y_test[i * group_size] for i in range(len(y_test) // group_size)]
        y_pred_grouped = [np.mean(y_pred[i * group_size:(i + 1) * group_size]) for i in range(len(y_test) // group_size)]
        auc = roc_auc_score(y_true, y_pred_grouped)
        print(f"{label} AUC: {auc:.4f}")
        nonlocal total_auc, auc_count
        total_auc += auc
        auc_count += 1
        return y_pred

    def model_multiary(X_train_seq, y_train, X_test_seq, y_test, num_classes, label):
        nonlocal total_auc, auc_count   #區域變數
        model = train_bilstm(X_train_seq, y_train, X_train_seq.shape[2], num_classes)
        torch.save(model.state_dict(), f"{label}_model.pth")
        probs = predict_bilstm_proba(model, X_test_seq)
        auc = roc_auc_score(y_test, probs, multi_class='ovr', average='micro')
        print(f"{label} AUC: {auc:.4f}")
        total_auc += auc
        auc_count += 1
       
        return probs
        

    gender_probs = model_binary(X_train_scaled, y_train_gender, X_test_scaled, y_test_gender, label="Gender")
    hold_probs = model_binary(X_train_scaled, y_train_hold, X_test_scaled, y_test_hold, label="Hold Hand")
    years_probs = model_multiary(X_train_seq, y_train_years, X_test_seq, y_test_years, len(np.unique(y_train_years)), label="Play Years")
    level_probs = model_multiary(X_train_seq, y_train_level, X_test_seq, y_test_level, len(np.unique(y_train_level)), label="Level")

    if auc_count > 0:
        final_score = total_auc / auc_count
        print(f"Final Score (Average AUC): {final_score:.4f}")

    final_results = []
    for i in range(len(id_test)):
        # Gender 預測
        gender_group = gender_probs[i * group_size:(i + 1) * group_size]
        gender_majority = int(np.mean(gender_group) >= 0.5)  # 判斷最常見的類別
        best_gender_prob = max([p for p in gender_group if (p >= 0.5) == (gender_majority == 1)], default=np.mean(gender_group))

        # Hold Hand 預測
        hold_group = hold_probs[i * group_size:(i + 1) * group_size]
        hold_majority = int(np.mean(hold_group) >= 0.5)
        best_hold_prob = max([p for p in hold_group if (p >= 0.5) == (hold_majority == 1)], default=np.mean(hold_group))

        row = [id_test[i], best_gender_prob, best_hold_prob]
        row += years_probs[i].tolist()  # 這裡已經是符合規範的機率分布
        row += level_probs[i].tolist()  # 同上
        final_results.append(row)

    colnames = ['unique_id', 'gender', 'hold_racket'] + \
               [f'playyears_{i}' for i in range(3)] + \
               [f'level_{i}' for i in range(2, 6)]

    df_out = pd.DataFrame(final_results, columns=colnames)
    df_out.to_csv('valid_prediction.csv', index=False, float_format="%.2f")
    print("預測完成，輸出檔為 valid_prediction.csv")


def predict_new_data(): #預測test_data

    group_size = 27
    info = pd.read_csv('test_info.csv')
    if not os.path.exists('./tabular_data_test'):
        test_data_generate()
    datapath = './tabular_data_test'  # 假設你把 test 特徵檔案放這裡
    datalist = list(Path(datapath).glob('**/*.csv'))
    
    x_test = pd.DataFrame()
    id_test = []

    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        data = pd.read_csv(file)
        #if data.empty:  # 跳過3211加這行！
           # print(f"Skip empty data file: {file}")
        # continue
        print(f"File: {file}, Data Shape: {data.shape}")
        #if not data.empty and not data.isna().all().all():
           # x_test = pd.concat([x_test, data], ignore_index=True)

        x_test = pd.concat([x_test, data], ignore_index=True)
        id_test.append(unique_id)

    # 注意：應該載入之前的 scaler，如果有的話
    scaler = joblib.load('scaler.pkl')
    X_test_scaled = scaler.transform(x_test)


    #序列格式

    input_dim = X_test_scaled.shape[1]
    X_test_seq = X_test_scaled.reshape(-1, group_size, input_dim)

    # 載入已經訓練好的模型
    gender_model = joblib.load('Gender_model.pkl')
    hold_model = joblib.load('Hold Hand_model.pkl')
    years_model = BiLSTMClassifier(input_dim=input_dim,cnn_channels=32, lstm_hidden_dim=64, transformer_dim=128, output_dim=3, num_heads=8, num_layers=2, dropout_rate=0.3).to(device)
    years_model.load_state_dict(torch.load('Play Years_model.pth', map_location=device))

    level_model = BiLSTMClassifier(input_dim=input_dim, cnn_channels=32, lstm_hidden_dim=64, transformer_dim=128, output_dim=4, num_heads=8, num_layers=2,  dropout_rate=0.3).to(device)
    level_model.load_state_dict(torch.load('Level_model.pth', map_location=device))

    group_size = 27  # 若測試資料有不同揮拍次數，要動態處理

    # 直接使用模型預測
    gender_probs = gender_model.predict_proba(X_test_scaled)[:, 1]
    hold_probs = hold_model.predict_proba(X_test_scaled)[:, 1]
    years_probs =  predict_bilstm_proba(years_model, X_test_seq)
    level_probs = predict_bilstm_proba(level_model, X_test_seq)
    final_results = []

    for i in range(len(id_test)):
        # Gender 預測
        gender_group = gender_probs[i * group_size:(i + 1) * group_size]
        gender_majority = int(np.mean(gender_group) >= 0.5)  # 判斷最常見的類別
        best_gender_prob = max([p for p in gender_group if (p >= 0.5) == (gender_majority == 1)], default=np.mean(gender_group))

        # Hold Hand 預測
        hold_group = hold_probs[i * group_size:(i + 1) * group_size]
        hold_majority = int(np.mean(hold_group) >= 0.5)
        best_hold_prob = max([p for p in hold_group if (p >= 0.5) == (hold_majority == 1)], default=np.mean(hold_group))

        row = [id_test[i], best_gender_prob, best_hold_prob]
        row += years_probs[i].tolist()   # 保證是 list 格式
        row += level_probs[i].tolist()   # 保證是 list 格式
        final_results.append(row)

    colnames = ['unique_id', 'gender', 'hold racket handed'] + \
               [f'play years_{i}' for i in range(3)] + \
               [f'level_{i}' for i in range(2, 6)]

    df_out = pd.DataFrame(final_results, columns=colnames)
    df_out.to_csv('test_prediction.csv', index=False, float_format="%.2f")
    print("預測完成，輸出檔為 test_prediction.csv")

if __name__ == "__main__":
    # 先強制執行訓練流程
    print("執行模型訓練...")
    main()

    #檢查模型檔案是否存在，確認已完成訓練
    required_models = ['Gender_model.pkl', 'Hold Hand_model.pkl', 
                       'Play Years_model.pkl', 'Level_model.pkl', 'scaler.pkl']
    if all(os.path.exists(model) for model in required_models):
        #訓練完成後詢問是否要進行預測
        mode = input("模型訓練完成，是否執行測試？(yes/no): ").strip().lower()
        if mode == "yes":
            predict_new_data()
        else:
            print("已完成訓練，未進行測試。")
    else:
        print("模型檔案不完整，請確認訓練是否成功。")
