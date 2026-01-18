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
def update_model_weights(model: Sequential, X_train: np.ndarray, y_train: np.ndarray, epochs: int=10, batch_size: int=32) -> Sequential:
    """

    Updates the weights of a pre-trained model using new training data.

    Parameters:

    - model (Sequential): The pre-trained Keras model.

    - X_train (np.ndarray): The new feature data for training.

    - y_train (np.ndarray): The new target data for training.

    - epochs (int): The number of training epochs (default is 10).

    - batch_size (int): The batch size for training (default is 32).

    Returns:

    - Sequential: The updated Keras model.

    """
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model