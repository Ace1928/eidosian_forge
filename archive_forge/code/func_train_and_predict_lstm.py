import pandas as pd  # For data manipulation and analysis
import requests  # For making HTTP requests to a specified URL
import pandas_ta as ta  # For technical analysis indicators
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
from termcolor import colored as cl  # For coloring terminal text
import math  # Provides access to mathematical functions
import logging  # For tracking events that happen when some software runs
import nltk  # For natural language processing tasks
from newspaper import Article  # For extracting and parsing news articles
import ccxt  # Cryptocurrency exchange library for connecting to various exchanges
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Union, Optional, Tuple, List, Dict, Any, Callable
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import gspread
from oauth2client.client import OAuth2Credentials
from tkinter import Tk, Label, Entry, Button, StringVar
import json
import os
import yfinance as yf
from itertools import product
from concurrent.futures import ThreadPoolExecutor, wait
import time
import yaml
import traceback
from typing import TypeAlias
@log_exception
@log_function_call
def train_and_predict_lstm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, epochs: int=100, batch_size: int=32) -> np.ndarray:
    """

    Trains an LSTM model on the provided training data and generates predictions on the test data.

    Parameters:

    - X_train (np.ndarray): The feature data used for training the model.

    - y_train (np.ndarray): The target data used for training the model.

    - X_test (np.ndarray): The feature data used for generating predictions.

    - epochs (int): The total number of training cycles (default is 100).

    - batch_size (int): The number of samples per gradient update (default is 32).

    Returns:

    - np.ndarray: An array containing the predicted values for the test data.

    """
    model = build_lstm_model(input_shape=X_train.shape[1:])
    early_stopping_callback = EarlyStopping(monitor='loss', patience=10, verbose=1)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping_callback], verbose=1)
    predictions = model.predict(X_test)
    return predictions