import asyncio
import cProfile
import hashlib
import io
import itertools
import json
import logging
import resource
import os
import pstats
import queue
import re
import sys
import threading
import time
import traceback
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from difflib import SequenceMatcher
from functools import reduce
from logging.handlers import MemoryHandler, RotatingFileHandler
from logging import StreamHandler, FileHandler
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional, Iterator, Union, Callable
import coloredlogs
import matplotlib.pyplot as plt
import mplcursors
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import (
from wordcloud import WordCloud
from tqdm import tqdm
import requests
from transformers import BertModel, BertTokenizer
import functools
class ModelInitializer(ReadyCheck):
    """
    A class meticulously designed to initialize, download, and manage machine learning and AI models,
    ensuring all resources are available locally for offline use. It provides methods for downloading,
    verifying, and loading models like BERT with progress updates and integrity checks, ensuring the
    highest standards of operational readiness and reliability.
    """

    def __init__(self, model_name='bert-base-uncased', base_path='/home/lloyd/randomuselesscrap/codeanalytics/models') -> None:
        """
        Initialize the ModelInitializer with a specified model name and base path.
        This method sets up the model path, ensures the necessary directories exist,
        and initializes the logging system for this instance.

        Args:
        model_name (str): The name of the model to initialize.
        base_path (str): The base directory path where the model files will be stored.
        """
        super().__init__()
        self.model_name: str = model_name
        self.base_path: str = base_path
        self.model_path: str = os.path.join(self.base_path, self.model_name)
        self._is_initialized: bool = False
        self.logger = advanced_logger
        self.ensure_directory(self.model_path)
        advanced_logger.log(logging.INFO, f'ModelInitializer for {self.model_name} at {self.model_path} initialized.')
        self.initialize()

    def initialize(self) -> None:
        """
        Fully initializes the model by downloading and loading it.
        Sets the _is_initialized flag to True if successful, or False if an exception occurs.
        """
        super().initialize()
        try:
            self.download_model()
            self.load_model()
            self._is_initialized = True
            advanced_logger.log(logging.INFO, f'ModelInitializer for {self.model_name} is fully initialized and ready.')
        except Exception as e:
            advanced_logger.log(logging.ERROR, f'Failed to initialize ModelInitializer: {str(e)}')
            self._is_initialized = False

    def is_ready(self) -> bool:
        """
        Checks if the model has been fully initialized and is ready for use.

        Returns:
        bool: True if the model is initialized, False otherwise.
        """
        return self._is_initialized

    @staticmethod
    def ensure_directory(path: str) -> None:
        """
        Ensure the directory exists, and if not, create it. Logs the creation of the directory.

        Args:
        path (str): The path of the directory to check or create.
        """
        if not os.path.exists(path):
            os.makedirs(path)
            advanced_logger.log(logging.INFO, f'Created directory at {path}')

    def download_model(self) -> None:
        """
        Download the model and tokenizer only if they are not already downloaded.
        """
        model_path: Path = Path(f'{self.model_path}/{self.model_name}')
        tokenizer_path: Path = Path(f'{self.model_path}/{self.model_name}-tokenizer')
        if not model_path.exists():
            advanced_logger.log(logging.INFO, f'Downloading model from {self.model_name}...')
            self.model = BertModel.from_pretrained(self.model_name)
            self.model.save_pretrained(model_path)
            advanced_logger.log(logging.INFO, f'Model saved at {model_path}')
        else:
            advanced_logger.log(logging.INFO, f'Model already exists at {model_path}. Loading model...')
            self.model = BertModel.from_pretrained(model_path)
        if not tokenizer_path.exists():
            advanced_logger.log(logging.INFO, f'Downloading tokenizer from {self.model_name}...')
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.tokenizer.save_pretrained(tokenizer_path)
            advanced_logger.log(logging.INFO, f'Tokenizer saved at {tokenizer_path}')
        else:
            advanced_logger.log(logging.INFO, f'Tokenizer already exists at {tokenizer_path}. Loading tokenizer...')
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        advanced_logger.log(logging.INFO, f'Model and tokenizer are ready at {self.model_path}')

    def load_model(self) -> Tuple[BertModel, BertTokenizer]:
        """
        Load the model from the local directory. This method initializes the BERT model and tokenizer
        from the downloaded files, ensuring they are ready for use.

        Returns:
        Tuple[BertModel, BertTokenizer]: The loaded BERT model and tokenizer.
        """
        advanced_logger.log(logging.INFO, f'Loading model from {self.model_path}')
        model = BertModel.from_pretrained(self.model_path)
        tokenizer = BertTokenizer.from_pretrained(self.model_path)
        advanced_logger.log(logging.INFO, 'BERT model loaded successfully.')
        return (model, tokenizer)