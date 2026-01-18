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
    filedialog,
    messagebox,
    simpledialog,
    ttk,
    scrolledtext,
    Menu,
    Spinbox,
    Label,
    Entry,
    Button,
    Text,
    END,
    HORIZONTAL,
    VERTICAL,
)
from wordcloud import WordCloud
from tqdm import tqdm
import requests
from transformers import BertModel, BertTokenizer
import functools


class ReadyCheck:
    def __init__(self):
        self._is_initialized = False

    def initialize(self):
        # This method should be overridden by subclasses to perform specific initializations
        self._is_initialized = True

    def is_ready(self):
        # Check if the class is initialized properly
        return self._is_initialized


class AdvancedLoggingSystem(ReadyCheck):
    """
    An exceptionally advanced, thread-safe, non-blocking, crash-resistant, and comprehensive logging, profiling, and error handling system meticulously designed to capture every conceivable detail of the application's operation in a secure, isolated, and prioritized manner. This system is capable of functioning in both synchronous and asynchronous environments with robust flexibility and dynamism.

    This class encapsulates the setup and management of a multi-threaded logging system that ensures:
    - Non-blocking logging operations to prevent interference with the main application flow.
    - Isolation in a separate thread to ensure logging operations are prioritized and secure.
    - Crash-resistance to capture logs even in the event of unexpected failures.
    - Detailed and comprehensive log entries for meticulous analysis including memory profiling, traceback details, and system state at the time of logging.
    - Integrated performance profiling to monitor and optimize the application's performance continuously.
    """

    def __init__(self):
        super().__init__()
        self.log_queue = asyncio.Queue()
        self.log_thread = threading.Thread(target=self.process_logs, daemon=True)
        self.profiler = cProfile.Profile()
        self.name = "AdvancedLoggingSystem"
        self.initialize()

    def initialize(self):
        super().initialize()
        self.setup_logging()
        self.log_thread.start()
        self._is_initialized = True
        self.is_ready()

    def setup_logging(self):
        """
        Configures the logging handlers, formatters, and levels to ensure detailed, comprehensive, and secure logging.
        """
        self.logger = logging.getLogger("AdvancedJSONProcessor")
        self.logger.setLevel(logging.DEBUG)  # Capture all levels of log messages

        # Enhanced formatter with colorized output for console readability
        formatter = coloredlogs.ColoredFormatter(
            "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            field_styles={
                "asctime": {"color": "green"},
                "levelname": {"bold": True, "color": "black"},
                "module": {"color": "blue"},
                "funcName": {"color": "cyan"},
                "lineno": {"color": "magenta"},
            },
        )

        # Setup handlers with advanced features
        self.setup_file_handler(formatter)
        self.setup_stream_handler(formatter)
        self.setup_memory_handler(formatter)

    def setup_file_handler(self, formatter):
        """
        Sets up file handlers with rotation, encoding, and profiling features for detailed file logging.
        """
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        for level_name, level in log_levels.items():
            file_handler = RotatingFileHandler(
                filename=f"{level_name.lower()}_processing.log",
                mode="a",
                maxBytes=100 * 1024 * 1024,  # 100MB
                backupCount=10,
                encoding="utf-8",
                delay=False,
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            self.logger.addHandler(file_handler)

        # Comprehensive log file handler
        comprehensive_file_handler = RotatingFileHandler(
            filename="comprehensive_processing.log",
            mode="a",
            maxBytes=100 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
            delay=False,
        )
        comprehensive_file_handler.setFormatter(formatter)
        comprehensive_file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(comprehensive_file_handler)

    def setup_stream_handler(self, formatter):
        """
        Sets up a stream handler for console output with detailed formatting.
        """
        stream_handler = StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(stream_handler)

    def setup_memory_handler(self, formatter):
        """
        Sets up a memory handler to buffer log entries and flush them conditionally to enhance performance.
        """
        comprehensive_file_handler = self.logger.handlers[
            1
        ]  # Assuming comprehensive handler is second
        memory_handler = MemoryHandler(
            capacity=100,
            flushLevel=logging.ERROR,
            target=comprehensive_file_handler,
        )
        memory_handler.setFormatter(formatter)
        memory_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(memory_handler)

    def process_logs(self):
        """
        Processes log messages from the queue, either synchronously or asynchronously, capturing detailed profiling, traceback information, and system state.
        This method ensures that it can be called in any context by dynamically managing the event loop and wrapping the asynchronous logic inside a synchronous method if needed.
        """

        async def async_process_logs():
            """
            Continuously processes log messages from the queue in a dedicated thread, capturing detailed profiling, traceback information, and system state.
            """
            while True:
                try:
                    record = await self.log_queue.get()
                    if record is None:
                        break  # Allows the thread to exit cleanly
                    if not self.profiler.is_running():
                        self.profiler.enable()
                    self.logger.handle(record)
                    self.profiler.disable()
                    s = io.StringIO()
                    sortby = pstats.SortKey.CUMULATIVE
                    ps = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
                    ps.print_stats()
                    logging.debug("Profile data:\n%s" % s.getvalue())
                    # Capture system state
                    system_state = {
                        "timestamp": datetime.now().isoformat(),
                        "system_load": os.getloadavg(),
                        "memory_usage": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss,
                    }
                    logging.debug(f"System state at log time: {system_state}")
                except Exception as e:
                    # Capture any exceptions and log them before exiting
                    exc_info = sys.exc_info()
                    traceback_details = {
                        "filename": exc_info[2].tb_frame.f_code.co_filename,
                        "lineno": exc_info[2].tb_lineno,
                        "name": exc_info[2].tb_frame.f_code.co_name,
                        "type": exc_info[0].__name__,
                        "message": str(e),
                    }
                    log_msg = (
                        "Logging thread encountered an exception: {details}".format(
                            details=traceback_details
                        )
                    )
                    self.logger.critical(log_msg, exc_info=True)
                    break

        # Check if the current function is being called in an asyncio context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop, setup a new one for the thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If the loop is already running, create a new task
            asyncio.create_task(async_process_logs())
        else:
            # Run the coroutine in the event loop
            loop.run_until_complete(async_process_logs())

    def log(self, level, msg):
        """
        Adds a log message to the queue to be processed by the logging thread, ensuring all details are captured including the current system state.
        """
        record = self.logger.makeRecord(
            self.logger.name, level, fn="", lno=0, msg=msg, args=None, exc_info=None
        )
        self.log_queue.put_nowait(record)


# Initialize the AdvancedLoggingSystem
advanced_logger = (
    AdvancedLoggingSystem()
)  # Future reference: advanced_logger.log(logging.INFO, "Logging system initialized and operational.")
advanced_logger.setup_logging()
advanced_logger.log(logging.INFO, "Logging system initialized and operational.")


class ModelInitializer(ReadyCheck):
    """
    A class meticulously designed to initialize, download, and manage machine learning and AI models,
    ensuring all resources are available locally for offline use. It provides methods for downloading,
    verifying, and loading models like BERT with progress updates and integrity checks, ensuring the
    highest standards of operational readiness and reliability.
    """

    def __init__(
        self,
        model_name="bert-base-uncased",
        base_path="/home/lloyd/randomuselesscrap/codeanalytics/models",
    ) -> None:
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
        self.logger = logging.getLogger(__name__)
        self.ensure_directory(self.model_path)
        advanced_logger.log(
            logging.INFO,
            f"ModelInitializer for {self.model_name} at {self.model_path} initialized.",
        )
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
            advanced_logger.log(
                logging.INFO,
                f"ModelInitializer for {self.model_name} is fully initialized and ready.",
            )
        except Exception as e:
            advanced_logger.log(
                logging.ERROR,
                f"Failed to initialize ModelInitializer: {str(e)}",
            )
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
            advanced_logger.log(logging.INFO, f"Created directory at {path}")

    def download_model(self) -> None:
        """
        Download all necessary files for the model if they do not exist or fail integrity checks.
        This method handles the downloading of multiple files required for the model, ensuring each
        is present and valid before proceeding.
        """
        advanced_logger.log(logging.INFO, "Downloading BERT model...")
        files_to_download = {
            "config.json": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
            "pytorch_model.bin": "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin",
            "vocab.txt": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        }
        for file_name, url in files_to_download.items():
            file_path = os.path.join(self.model_path, file_name)
            if not os.path.exists(file_path) or not self.verify_file(
                file_path, "expected_hash_placeholder"
            ):
                advanced_logger.log(logging.INFO, f"Downloading {file_name}...")
                self.download_file(url, file_path)
                if not self.verify_file(file_path, "expected_hash_placeholder"):
                    raise Exception(f"Failed to verify {file_name}")
        advanced_logger.log(logging.INFO, "BERT model downloaded successfully.")

    def load_model(self) -> Tuple[BertModel, BertTokenizer]:
        """
        Load the model from the local directory. This method initializes the BERT model and tokenizer
        from the downloaded files, ensuring they are ready for use.

        Returns:
        Tuple[BertModel, BertTokenizer]: The loaded BERT model and tokenizer.
        """
        advanced_logger.log(logging.INFO, f"Loading model from {self.model_path}")
        model = BertModel.from_pretrained(self.model_path)
        tokenizer = BertTokenizer.from_pretrained(self.model_path)
        advanced_logger.log(logging.INFO, "BERT model loaded successfully.")
        return model, tokenizer


# Ensure NLTK resources are downloaded
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("vader_lexicon")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")


class UniversalDataManager(ReadyCheck):
    """
    A class meticulously designed to manage, store, and retrieve data across various components of the advanced program.
    It acts as a central repository manager, ensuring robust, universal, and dynamic data management, seamlessly integrating with
    the AdvancedLoggingSystem, ModelInitializer, AdvancedJSONProcessor, and ModernGUI for comprehensive, optimized, and efficient data handling.
    This class is the epitome of data management excellence, embodying flexibility, functionality, and extensibility.
    """

    def __init__(self, repository_path: str):
        """
        Initialize the UniversalDataManager with a specified repository path, ensuring it is ready for immediate and robust data management.
        This initialization also prepares the system for seamless integration with other system components such as AdvancedLoggingSystem,
        ModelInitializer, AdvancedJSONProcessor, and ModernGUI, ensuring a cohesive and efficient operation.
        """
        super().__init__()
        self.repository_path = Path(repository_path)
        self.ensure_repository_directory()
        self.initialize_system_components()
        self.cache = (
            {}
        )  # Initialize an empty cache dictionary for storing frequently accessed data
        advanced_logger.log(
            logging.INFO,
            f"UniversalDataManager initialized with repository at {self.repository_path}",
        )

    def initialize(self):
        super().initialize()
        self.ensure_repository_directory()
        self._is_initialized = os.path.exists(self.repository_path)

    def ensure_repository_directory(self) -> None:
        """
        Ensure that the repository directory exists; if not, create it, ensuring robustness and readiness for data storage and retrieval.
        This method is crucial for maintaining the integrity and availability of the data managed by the UniversalDataManager.
        """
        if not self.repository_path.exists():
            self.repository_path.mkdir(parents=True, exist_ok=True)
            advanced_logger.log(
                logging.INFO,
                f"Created repository directory at {self.repository_path}",
            )

    def initialize_system_components(self) -> None:
        """
        Initialize and integrate system components required for the UniversalDataManager to function optimally.
        This includes setting up connections with the AdvancedLoggingSystem, ModelInitializer, AdvancedJSONProcessor, and ModernGUI.
        Each component is initialized and configured to work in harmony with the UniversalDataManager, ensuring a robust and efficient data management system.
        """
        # Integration with AdvancedLoggingSystem
        if not hasattr(self, "logger"):
            self.logger = advanced_logger

        # Integration with ModelInitializer
        if not hasattr(self, "model_initializer"):
            self.model_initializer = ModelInitializer()

        # Integration with AdvancedJSONProcessor
        if not hasattr(self, "json_processor"):
            self.json_processor = AdvancedJSONProcessor(
                file_path="", repository_path=self.repository_path
            )

        # Integration with ModernGUI
        if not hasattr(self, "gui"):
            self.gui = ModernGUI()

        self.logger.log(
            logging.INFO,
            "UniversalDataManager is now fully integrated and operational with all system components.",
        )

    def cache_data(
        self, file_name: str, data: Dict[str, Any], expiration_time: int = 3600
    ) -> None:
        """
        Cache data with an expiration time to reduce I/O operations and improve performance.
        """
        self.cache[file_name] = {
            "data": data,
            "expiration_time": time.time() + expiration_time,
        }
        self.logger.log(
            logging.INFO,
            f"Data cached for file: {file_name}",
        )

    def get_cached_data(self, file_name: str) -> Dict[str, Any]:
        """
        Retrieve cached data if available and not expired.
        """
        if file_name in self.cache:
            if self.cache[file_name]["expiration_time"] > time.time():
                self.logger.log(
                    logging.INFO,
                    f"Retrieved cached data for file: {file_name}",
                )
                return self.cache[file_name]["data"]
            else:
                del self.cache[file_name]
                self.logger.log(
                    logging.INFO,
                    f"Cache expired for file: {file_name}",
                )
        return None

    def store_data(self, data: Dict[str, Any], file_name: str) -> None:
        """
        Store data in a specified file within the repository path, ensuring data integrity and availability for future retrieval.
        """
        file_path = self.repository_path / file_name
        try:
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)
            self.cache_data(file_name, data)  # Cache the stored data
            advanced_logger.log(
                logging.INFO,
                f"Data stored successfully at {file_path}",
            )
        except IOError as io_error:
            advanced_logger.log(
                logging.ERROR,
                f"IO error storing data at {file_path}: {io_error}",
            )
            raise IOError(f"IO error at {file_path}: {io_error}") from io_error
        except json.JSONDecodeError as json_error:
            advanced_logger.log(
                logging.ERROR,
                f"JSON encoding error at {file_path}: {json_error}",
            )
            raise json.JSONDecodeError(
                f"JSON encoding error at {file_path}: {json_error}"
            ) from json_error
        except Exception as e:
            advanced_logger.log(
                logging.ERROR,
                f"Unexpected error storing data at {file_path}: {e}",
            )
            raise Exception(f"Unexpected error at {file_path}: {e}") from e

    def retrieve_data(self, file_name: str) -> Dict[str, Any]:
        """
        Retrieve data from a specified file within the repository path, ensuring data is accurately and efficiently fetched.
        """
        file_path = self.repository_path / file_name
        cached_data = self.get_cached_data(file_name)
        if cached_data:
            return cached_data
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            self.cache_data(file_name, data)  # Cache the retrieved data
            self.logger.log(
                logging.INFO,
                f"Data retrieved successfully from {file_path}",
            )
            return data
        except FileNotFoundError as fnf_error:
            self.logger.log(
                logging.ERROR,
                f"File not found at {file_path}: {fnf_error}",
            )
            raise FileNotFoundError(
                f"File not found at {file_path}: {fnf_error}"
            ) from fnf_error
        except json.JSONDecodeError as json_error:
            self.logger.log(
                logging.ERROR,
                f"JSON decoding error at {file_path}: {json_error}",
            )
            raise json.JSONDecodeError(
                f"JSON decoding error at {file_path}: {json_error}"
            ) from json_error
        except Exception as e:
            self.logger.log(
                logging.ERROR,
                f"Unexpected error retrieving data from {file_path}: {e}",
            )
            raise Exception(
                f"Unexpected error retrieving data from {file_path}: {e}"
            ) from e

    def load_session_data(self):
        """
        Load session data from a persistent storage to maintain state across different runs of the application.
        """
        session_file_path = self.repository_path / "session_data.json"
        try:
            with open(session_file_path, "r") as file:
                self.session_data = json.load(file)
            advanced_logger.log(
                logging.INFO,
                f"Session data loaded successfully from {session_file_path}",
            )
        except FileNotFoundError:
            self.session_data = {}
            advanced_logger.log(
                logging.WARNING,
                f"No existing session data found. Starting with an empty state.",
            )

    def save_session_data(self):
        """
        Save current session data to a file to maintain state across different runs of the application.
        """
        session_file_path = self.repository_path / "session_data.json"
        try:
            with open(session_file_path, "w") as file:
                json.dump(self.session_data, file, indent=4)
            advanced_logger.log(
                logging.INFO,
                f"Session data saved successfully at {session_file_path}",
            )
        except Exception as e:
            advanced_logger.log(
                logging.ERROR,
                f"Error saving session data at {session_file_path}: {e}",
            )
            raise Exception(
                f"Session data saving error at {session_file_path}: {e}"
            ) from e

    def update_session_data(self, key: str, value: Any):
        """
        Update the session data with new information, ensuring data consistency and availability.
        """
        self.session_data[key] = value
        self.save_session_data()
        advanced_logger.log(
            logging.INFO,
            f"Session data updated for key {key}.",
        )


class AdvancedJSONProcessor(UniversalDataManager):
    """
    An advanced class to process JSON data with complex, flexible, robust, intelligent, and comprehensive error handling.
    This class is designed to handle JSON data processing with an emphasis on meticulous detail, sophisticated logic,
    and a robust architecture that ensures high reliability and performance.

    Attributes:
        file_path (str): The path of the JSON file to process.
        data (Dict[str, Any]): The parsed JSON data, initialized as None and populated by read_json method.
        repository_path (Path): Path to the central repository where all processed data is stored.
        cache (Dict[str, Any]): A cache to store frequently accessed data to reduce I/O operations.
    """

    def __init__(self, file_path: str, repository_path: str):
        super().__init__(repository_path)
        self.file_path: str = file_path
        self.data: Optional[Dict[str, Any]] = None
        self._is_initialized: bool = False

    def initialize(self) -> None:
        """
        Initialize the JSON processor by reading the JSON file, logging the initialization process, and ensuring the repository directory exists.
        This method has been enhanced with specific error handling for FileNotFoundError, json.JSONDecodeError, and general exceptions to provide detailed feedback and ensure robust initialization.
        """
        super().load_session_data()
        try:
            if not self.repository_path.exists():
                self.repository_path.mkdir(parents=True, exist_ok=True)
            self.data = self.read_json(self.file_path)
            advanced_logger.log(
                logging.INFO,
                f"AdvancedJSONProcessor initialized with file path: {self.file_path}",
            )
        except FileNotFoundError as fnf_error:
            advanced_logger.log(
                logging.ERROR,
                f"File not found during initialization: {fnf_error}",
            )
            raise FileNotFoundError(f"File not found: {fnf_error}") from fnf_error
        except json.JSONDecodeError as json_error:
            advanced_logger.log(
                logging.ERROR,
                f"JSON decoding error during initialization: {json_error}",
            )
            raise json.JSONDecodeError(
                f"JSON decoding error: {json_error}"
            ) from json_error
        except Exception as e:
            advanced_logger.log(
                logging.ERROR,
                f"Unexpected error during initialization: {e}",
            )
            raise Exception(f"Unexpected error: {e}") from e
        self._is_initialized = True
        self.is_ready()

    def is_ready(self) -> bool:
        """
        Check if the processor has been successfully initialized and is ready to process data.

        Returns:
            bool: True if initialized, False otherwise.
        """
        return self._is_initialized

    def read_json(self, file_path: str) -> Dict[str, Any]:
        """
        Reads a JSON file and returns the content as a dictionary with robust error handling.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            Dict[str, Any]: Parsed JSON content as a dictionary.

        Raises:
            FileNotFoundError: If the JSON file is not found at the specified path.
            json.JSONDecodeError: If the JSON file is not properly formatted.
            Exception: For any other unexpected errors that may occur.
        """
        data = self.retrieve_data(file_path)
        if data:
            return data

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                self.cache_data(file_path, data)
                advanced_logger.log(
                    logging.DEBUG,
                    f"Successfully read JSON data from {file_path} and stored in cache.",
                )
                return data
        except FileNotFoundError as fnf_error:
            advanced_logger.log(
                logging.ERROR,
                f"File not found during JSON reading: {fnf_error}",
            )
            raise FileNotFoundError(f"File not found: {fnf_error}") from fnf_error
        except json.JSONDecodeError as json_error:
            advanced_logger.log(
                logging.ERROR,
                f"JSON decoding error during JSON reading: {json_error}",
            )
            raise json.JSONDecodeError(
                f"JSON decoding error: {json_error}"
            ) from json_error
        except Exception as e:
            advanced_logger.log(
                logging.ERROR,
                f"Unexpected error during JSON reading: {e}",
            )
            raise Exception(f"Unexpected error: {e}") from e

    def clean_text(self, text: str) -> Dict[str, Any]:
        """
        Cleanses and enriches the input text using advanced natural language processing techniques.

        Args:
            text (str): The text to be cleaned.

        Returns:
            Dict[str, Any]: A dictionary containing the cleaned text and relevant metadata.

        Raises:
            FileNotFoundError: If the JSON file is not found at the specified path.
            json.JSONDecodeError: If the JSON file is not properly formatted.
            nltk.NLTKError: For errors related to NLTK processing.
            Exception: For any other unexpected errors that may occur during text cleaning.
        """
        cached_data = self.get_cached_data(text)
        if cached_data:
            return cached_data

        try:
            if not self.repository_path.exists():
                self.repository_path.mkdir(parents=True, exist_ok=True)
            self.data = self.read_json(self.file_path)
            advanced_logger.log(
                logging.INFO,
                f"AdvancedJSONProcessor initialized with file path: {self.file_path}",
            )

            refined_regex_pattern: str = r"[^a-zA-Z.,'\s]+"
            cleaned_text: str = re.sub(refined_regex_pattern, "", text)

            tokens: List[str] = word_tokenize(cleaned_text)
            filtered_tokens: List[str] = [
                word
                for word in tokens
                if word.lower() not in stopwords.words("english")
            ]

            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens: List[str] = [
                lemmatizer.lemmatize(token) for token in filtered_tokens
            ]

            advanced_cleaned_text: str = " ".join(lemmatized_tokens)

            sentiment_analyzer = SentimentIntensityAnalyzer()
            sentiment_score = sentiment_analyzer.polarity_scores(advanced_cleaned_text)

            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased")

            inputs = tokenizer(advanced_cleaned_text, return_tensors="pt")
            outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state
            enhanced_text_vector = last_hidden_states.mean(dim=1).squeeze().tolist()

            result = {
                "cleaned_text": advanced_cleaned_text,
                "sentiment_analysis": sentiment_score,
                "contextual_embeddings": enhanced_text_vector,
                "lemmatized_tokens": lemmatized_tokens,
                "filtered_tokens": filtered_tokens,
            }

            self.cache_data(text, result)
            advanced_logger.log(
                logging.INFO,
                f"Text cleaning and enrichment completed and stored in cache: {result}",
            )

            return result
        except FileNotFoundError as fnf_error:
            advanced_logger.log(
                logging.ERROR,
                f"File not found during text cleaning: {fnf_error}",
            )
            raise FileNotFoundError(f"File not found: {fnf_error}") from fnf_error
        except json.JSONDecodeError as json_error:
            advanced_logger.log(
                logging.ERROR,
                f"JSON decoding error during text cleaning: {json_error}",
            )
            raise json.JSONDecodeError(
                f"JSON decoding error: {json_error}"
            ) from json_error
        except nltk.NLTKError as nltk_error:
            advanced_logger.log(
                logging.ERROR,
                f"NLTK error in text cleaning: {nltk_error}",
            )
            raise nltk.NLTKError(f"NLTK processing error: {nltk_error}") from nltk_error
        except Exception as e:
            advanced_logger.log(
                logging.ERROR,
                f"General error in text cleaning: {e}",
            )
            raise Exception(f"General text processing error: {e}") from e

    def extract_text(
        self, data: Union[Dict[str, Any], List[Any], str]
    ) -> Dict[str, Any]:
        """
        Extracts text from JSON data structures with precision and efficiency, handling various data types and structures.

        Args:
            data (Union[Dict[str, Any], List[Any], str]): The JSON data to be processed.

        Returns:
            Dict[str, Any]: A dictionary containing all cleaned text entries, meticulously extracted and processed.
        """
        try:
            text_data: Dict[str, Any] = {}

            if isinstance(data, dict):
                for key, value in data.items():
                    text_data[key] = self.extract_text(value)
                    advanced_logger.log(
                        logging.DEBUG,
                        f"Extracted text from dictionary key: {key}",
                    )

            elif isinstance(data, list):
                for index, item in enumerate(data):
                    text_data[f"item_{index}"] = self.extract_text(item)
                    advanced_logger.log(
                        logging.DEBUG,
                        f"Extracted text from list item at index: {index}",
                    )

            elif isinstance(data, str):
                text_data["processed_text"] = self.clean_text(data)
                advanced_logger.log(
                    logging.DEBUG,
                    f"Cleaned and processed text: {data[:50]}...",
                )

            return text_data

        except FileNotFoundError as fnf_error:
            advanced_logger.log(
                logging.ERROR,
                f"File not found during text extraction: {fnf_error}",
            )
            raise FileNotFoundError(
                f"File not found during text extraction: {fnf_error}"
            ) from fnf_error

        except json.JSONDecodeError as json_error:
            advanced_logger.log(
                logging.ERROR,
                f"JSON decoding error during text extraction: {json_error}",
            )
            raise json.JSONDecodeError(
                f"JSON decoding error during text extraction: {json_error}"
            ) from json_error

        except nltk.NLTKError as nltk_error:
            advanced_logger.log(
                logging.ERROR,
                f"NLTK error during text extraction: {nltk_error}",
            )
            raise nltk.NLTKError(
                f"NLTK processing error during text extraction: {nltk_error}"
            ) from nltk_error

        except Exception as e:
            advanced_logger.log(
                logging.ERROR,
                f"Unexpected error during text extraction: {e}",
            )
            raise Exception(f"Unexpected error during text extraction: {e}") from e

    def process_data(
        self,
    ) -> Tuple[Set[str], Dict[str, List[str]], Dict[str, Set[str]], Dict[str, Any]]:
        """
        Processes the JSON data to find duplicates, map titles to IDs, assess similarity, and extract metadata.

        Returns:
            Tuple[Set[str], Dict[str, List[str]], Dict[str, Set[str]], Dict[str, Any]]: Processed data.
        """
        try:
            if not self.repository_path.exists():
                self.repository_path.mkdir(parents=True, exist_ok=True)
            if not self.data:
                self.data = self.read_json(self.file_path)
                advanced_logger.log(
                    logging.INFO,
                    f"AdvancedJSONProcessor initialized with file path: {self.file_path}",
                )

            entries: List[Dict[str, Any]] = self.data.get("entries", [])
            title_to_ids: Dict[str, List[str]] = defaultdict(list)
            duplicates: Set[str] = set()
            title_similarity: Dict[str, Set[str]] = defaultdict(set)
            metadata: Dict[str, Any] = defaultdict(dict)

            all_text_data = self.extract_text(self.data)

            vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
            tfidf_matrix = vectorizer.fit_transform(
                [text["cleaned_text"] for text in all_text_data.values()]
            )

            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            descriptions = [
                entry.get("description", "")
                for entry in entries
                if "description" in entry
            ]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(tfidf_matrix.toarray())
            db = DBSCAN(eps=0.3, min_samples=2).fit(X_scaled)
            labels = db.labels_

            cluster_dict = defaultdict(list)
            for idx, label in enumerate(labels):
                if idx < len(entries):
                    cluster_dict[label].append(entries[idx])

            for cluster, items in cluster_dict.items():
                if len(items) > 1:
                    for item in items:
                        title = item.get("title", "")
                        duplicates.add(title)
                        for other_item in items:
                            if other_item != item:
                                title_similarity[title].add(other_item.get("title", ""))

            for idx, entry in enumerate(entries):
                title = entry.get("title", "")
                id = entry.get("id", "")
                if title:
                    if title in title_to_ids:
                        if id not in title_to_ids[title]:
                            title_to_ids[title].append(id)
                        else:
                            duplicates.add(title)
                    else:
                        title_to_ids[title] = [id]

                    similar_indices = np.where(cosine_sim[idx] > 0.8)[0]
                    for index in similar_indices:
                        if index != idx:
                            similar_title = entries[index].get("title", "")
                            title_similarity[title].add(similar_title)
                            title_similarity[similar_title].add(title)

                    metadata[title].update(
                        {
                            "description": entry.get("description", ""),
                            "examples": entry.get("examples", []),
                            "related_standards": entry.get("related_standards", []),
                        }
                    )

            processed_data = {
                "duplicates": duplicates,
                "title_to_ids": dict(title_to_ids),
                "title_similarity": dict(title_similarity),
                "metadata": dict(metadata),
            }
            self.store_processed_data(processed_data, "processed_data.pkl")
            advanced_logger.log(
                logging.INFO, f"Processed data saved to processed_data.pkl"
            )

            return (
                duplicates,
                dict(title_to_ids),
                dict(title_similarity),
                dict(metadata),
            )
        except FileNotFoundError as fnf_error:
            advanced_logger.log(
                logging.ERROR,
                f"File not found during data processing: {fnf_error}",
            )
            raise FileNotFoundError(
                f"File not found during data processing: {fnf_error}"
            ) from fnf_error
        except json.JSONDecodeError as json_error:
            advanced_logger.log(
                logging.ERROR,
                f"JSON decoding error during data processing: {json_error}",
            )
            raise json.JSONDecodeError(
                f"JSON decoding error during data processing: {json_error}"
            ) from json_error
        except Exception as e:
            advanced_logger.log(
                logging.ERROR,
                f"Unexpected error during data processing: {e}",
            )
            raise Exception(f"Unexpected error during data processing: {e}") from e


class ModernGUI(ReadyCheck):
    """
    A sophisticated GUI class designed for file browsing, data processing, and visualization. This class integrates advanced features, comprehensive logging, responsive interactive elements, and a new 'Process' button to initiate data processing using AdvancedJSONProcessor. It also includes options for visualizing data through knowledge graphs and word clouds, allowing the user to dynamically interact and save visualizations based on their current view.

    Attributes:
        root (tk.Tk): The root window of the GUI.
        file_path (str): The path of the selected file.
        operation_status (tk.StringVar): Status of the current operation.
        thread_queue (queue.Queue): Queue for threading tasks.
        process_button (tk.Button): Button to initiate processing of selected file.
        visualize_button (tk.Button): Button to initiate visualization of processed data.
        data_manager (UniversalDataManager): Instance of UniversalDataManager for data management.
        json_processor (AdvancedJSONProcessor): Instance of AdvancedJSONProcessor for data processing.
    """

    def __init__(self):
        super().__init__()
        self.root = tk.Tk()
        self.root.title("Advanced Data Processor and Visualizer")
        self.root.geometry("1200x800")
        self.root.config(bg="lightblue")
        self.file_path = ""
        self.operation_status = tk.StringVar(value="Ready")
        self.thread_queue = queue.Queue()
        self.data_manager = UniversalDataManager(
            repository_path="/home/lloyd/randomuselesscrap/codeanalytics/repository"
        )
        self.json_processor = AdvancedJSONProcessor(
            file_path="",
            repository_path="/home/lloyd/randomuselesscrap/codeanalytics/repository",
        )
        self.setup_menu()
        self.setup_status_bar()
        self.setup_threading()
        self.setup_process_button()
        self.setup_visualize_button()
        advanced_logger.log(
            logging.INFO,
            "ModernGUI initialized with a visually appealing, functional, and interactive root window.",
        )

    def initialize(self):
        """
        Initialize the GUI by setting up the menu, status bar, threading, process button, and visualize button.
        """
        super().initialize()
        self.setup_menu()
        self.setup_status_bar()
        self.setup_threading()
        self.setup_process_button()
        self.setup_visualize_button()
        advanced_logger.log(
            logging.INFO,
            "ModernGUI initialized with a visually appealing, functional, and interactive root window.",
        )
        self.is_ready()

    def setup_menu(self):
        """
        Sets up the menu for the GUI, providing options for file operations, processing, visualization, and exit, enhanced with modern aesthetics.
        """
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(
            label="Open", command=lambda: self.thread_action(self.file_browse)
        )
        file_menu.add_command(
            label="Save", command=lambda: self.thread_action(self.file_save)
        )
        file_menu.add_command(
            label="Open Multiple",
            command=lambda: self.thread_action(self.file_browse_multiple),
        )
        file_menu.add_command(
            label="Open Directory",
            command=lambda: self.thread_action(self.directory_browse),
        )
        file_menu.add_command(
            label="Save Session", command=self.data_manager.save_session_data
        )
        file_menu.add_command(
            label="Load Session", command=self.data_manager.load_session_data
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menu_bar)
        advanced_logger.log(
            logging.DEBUG,
            "Menu setup completed with file operations, processing, visualization, and exit functionality.",
        )

    def setup_status_bar(self):
        """
        Sets up a status bar at the bottom of the GUI window to display the current operation status.
        """
        status_bar = tk.Label(
            self.root,
            textvariable=self.operation_status,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        advanced_logger.log(
            logging.DEBUG,
            "Status bar setup completed to display operation status.",
        )

    def setup_threading(self):
        """
        Sets up threading to handle file operations, processing, and visualization without freezing the GUI.
        """
        thread = threading.Thread(target=self.process_queue, daemon=True)
        thread.start()
        advanced_logger.log(
            logging.INFO,
            "Threading setup completed for efficient execution of tasks.",
        )

    def setup_process_button(self):
        """
        Sets up a 'Process' button that, when clicked, initiates the processing of the selected file using AdvancedJSONProcessor.
        """
        self.process_button = tk.Button(
            self.root,
            text="Process",
            command=self.process_data,
            bg="lightgreen",
            fg="black",
            font=("Helvetica", 12),
        )
        self.process_button.pack(pady=20)
        advanced_logger.log(
            logging.DEBUG,
            "Process button setup completed to initiate file processing.",
        )

    def setup_visualize_button(self):
        """
        Sets up a 'Visualize' button that, when clicked, initiates the visualization of processed data through knowledge graphs and word clouds.
        """
        self.visualize_button = tk.Button(
            self.root,
            text="Visualize",
            command=self.visualize_data,
            bg="lightblue",
            fg="black",
            font=("Helvetica", 12),
        )
        self.visualize_button.pack(pady=20)
        advanced_logger.log(
            logging.DEBUG,
            "Visualize button setup completed to initiate data visualization.",
        )

    def process_data(self):
        """
        Initiates the processing of the selected file using AdvancedJSONProcessor.
        """
        if self.file_path:
            try:
                self.json_processor.initialize_processor()
                processed_data = self.json_processor.process_data()
                self.data_manager.store_data(processed_data, "processed_data.pkl")
                self.operation_status.set("Data processing completed successfully.")
                advanced_logger.log(
                    logging.INFO,
                    f"Data processing completed successfully for {self.file_path}",
                )
            except Exception as e:
                self.operation_status.set(f"Error during data processing: {str(e)}")
                advanced_logger.log(
                    logging.ERROR, f"Data processing failed for {self.file_path}: {e}"
                )
                messagebox.showerror("Processing Error", f"An error occurred: {e}")
        else:
            self.operation_status.set("No file selected for processing.")
            advanced_logger.log(
                logging.INFO, "Data processing attempted without a file selected."
            )

    def visualize_data(self):
        """
        Initiates the visualization of processed data through knowledge graphs and word clouds, allowing dynamic interaction and saving of visualizations.
        """
        processed_data = self.data_manager.retrieve_data("processed_data.pkl")
        if processed_data:
            try:
                self.operation_status.set("Visualization started.")
                advanced_logger.log(
                    logging.INFO, f"Visualization started for processed data"
                )
                visualizer = DataVisualizer(processed_data)
                visualizer.visualize_data()
                self.operation_status.set("Visualization completed successfully.")
                advanced_logger.log(
                    logging.INFO,
                    f"Visualization completed successfully for processed data",
                )
            except Exception as e:
                self.operation_status.set("Visualization failed.")
                advanced_logger.log(
                    logging.ERROR, f"Visualization failed for processed data: {e}"
                )
                messagebox.showerror("Visualization Error", f"An error occurred: {e}")
        else:
            self.operation_status.set("No processed data found for visualization.")
            advanced_logger.log(
                logging.INFO,
                "Visualization attempted without processed data available.",
            )

    def process_queue(self):
        """
        Process tasks in the queue.
        """
        while True:
            try:
                function, args, kwargs = self.thread_queue.get(block=False)
                function(*args, **kwargs)
                self.thread_queue.task_done()
            except queue.Empty:
                continue

    def thread_action(self, func, *args, **kwargs):
        """
        Place an action in the queue to be processed by the threading system.
        """
        self.thread_queue.put((func, args, kwargs))

    def file_browse(self) -> str:
        """
        Browse for a file to open with an advanced file dialog, providing detailed logging and updating the status bar.

        Returns:
            str: The path of the selected file.
        """
        advanced_logger.log(logging.DEBUG, "Initiating browsing for a single file.")
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path = file_path
            self.data_manager.update_session_data("selected_file", file_path)
            self.json_processor.file_path = file_path
            self.operation_status.set(f"File selected: {file_path}")
            advanced_logger.log(logging.INFO, f"File selected: {file_path}")
        else:
            self.operation_status.set("No file selected.")
            advanced_logger.log(logging.INFO, "File browsing cancelled.")
        return file_path

    def file_save(self) -> str:
        """
        Browse for a location to save a file with an advanced file dialog, providing detailed logging and updating the status bar.

        Returns:
            str: The path of the selected location.
        """
        advanced_logger.log(
            logging.DEBUG, "Initiating browsing for a file save location."
        )
        save_path = filedialog.asksaveasfilename()
        if save_path:
            advanced_logger.log(logging.INFO, f"Save location selected: {save_path}")
            self.operation_status.set("Save location selected: " + save_path)
        else:
            advanced_logger.log(logging.INFO, "Save operation cancelled.")
            self.operation_status.set("Save operation cancelled.")
        return save_path

    def file_browse_multiple(self) -> List[str]:
        """
        Browse for multiple files to open with an advanced file dialog, providing detailed logging and updating the status bar.

        Returns:
            List[str]: The paths of the selected files.
        """
        advanced_logger.log(logging.DEBUG, "Initiating browsing for multiple files.")
        files = filedialog.askopenfilenames()
        if files:
            advanced_logger.log(logging.INFO, f"Multiple files selected: {files}")
            self.operation_status.set("Multiple files selected: " + str(files))
        else:
            advanced_logger.log(logging.INFO, "Multiple file browsing cancelled.")
            self.operation_status.set("Multiple file browsing cancelled.")
        return list(files)

    def directory_browse(self) -> str:
        """
        Browse for a directory to open with an advanced file dialog, providing detailed logging and updating the status bar.

        Returns:
            str: The path of the selected directory.
        """
        advanced_logger.log(logging.DEBUG, "Initiating browsing for a directory.")
        directory = filedialog.askdirectory()
        if directory:
            advanced_logger.log(logging.INFO, f"Directory selected: {directory}")
            self.operation_status.set("Directory selected: " + directory)
        else:
            advanced_logger.log(logging.INFO, "Directory browsing cancelled.")
            self.operation_status.set("Directory browsing cancelled.")
        return directory

    def run(self):
        """
        Run the GUI application with a modern aesthetic, providing detailed logging, responsive interaction, and a status bar for real-time updates.
        """
        advanced_logger.log(logging.INFO, "Launching the GUI application.")
        self.root.mainloop()

    def __del__(self):
        self.root.destroy()
        advanced_logger.log(logging.INFO, "GUI root window destroyed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.root.destroy()
        advanced_logger.log(
            logging.INFO, "Exited ModernGUI context and destroyed the root window."
        )


class DataVisualizer(ReadyCheck):
    """
    A comprehensive, flexible, and interactive data visualization class designed to work seamlessly with the ModernGUI, AdvancedJSONProcessor, and UniversalDataManager classes. This class provides robust and dynamic visualization capabilities for all processed data, metadata, and analysis results, ensuring maximum flexibility and customization options for the user.

    The DataVisualizer class leverages advanced plotting libraries such as Matplotlib, Seaborn, and Plotly to create visually appealing and informative visualizations. It supports a wide range of chart types, including line plots, bar charts, scatter plots, heatmaps, and interactive 3D plots.

    Key features:
    - Seamless integration with the existing application architecture
    - Support for various data formats and structures
    - Wide range of customizable chart types and styles
    - Interactive plot controls for zooming, panning, and selecting data points
    - Ability to save and export visualizations in multiple formats
    - Detailed logging and error handling using the AdvancedLoggingSystem
    """

    def __init__(
        self,
        data: Dict[str, Any],
        model_initializer: ModelInitializer,
        data_manager: UniversalDataManager,
    ):
        """
        Initialize the DataVisualizer with the processed data, model initializer, and data manager instances.

        Args:
            data (Dict[str, Any]): The processed data to be visualized.
            model_initializer (ModelInitializer): The initialized model for advanced visualizations.
            data_manager (UniversalDataManager): The data manager for accessing and storing visualization data.
        """
        super().__init__()
        self.data = data
        self.model_initializer = model_initializer
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        advanced_logger.log(
            logging.INFO,
            "DataVisualizer initialized with processed data, model initializer, and data manager.",
        )

    def initialize(self):
        super().initialize()
        self._is_initialized = True
        self.is_ready()

    def visualize_data(
        self,
        chart_type: str,
        x_axis: str,
        y_axis: str,
        z_axis: Optional[str] = None,
        color: Optional[str] = None,
        style: Optional[str] = None,
        interactive: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize the processed data using the specified chart type and customization options.

        Args:
            chart_type (str): The type of chart to create (e.g., 'line', 'bar', 'scatter', 'heatmap', '3d').
            x_axis (str): The data column to use for the x-axis.
            y_axis (str): The data column to use for the y-axis.
            z_axis (Optional[str]): The data column to use for the z-axis (for 3D plots).
            color (Optional[str]): The data column to use for color encoding.
            style (Optional[str]): The style theme for the chart (e.g., 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks').
            interactive (bool): Whether to enable interactive plot controls.
            save_path (Optional[str]): The path to save the visualization, if desired.

        Returns:
            None
        """
        try:
            # Validate input data and options
            self._validate_data(x_axis, y_axis, z_axis, color)
            self._validate_options(chart_type, style, interactive)

            # Prepare data for visualization
            x_data, y_data, z_data, color_data = self._prepare_data(
                x_axis, y_axis, z_axis, color
            )

            # Create the specified chart type
            chart_creator = {
                "line": self._create_line_plot,
                "bar": self._create_bar_chart,
                "scatter": self._create_scatter_plot,
                "heatmap": self._create_heatmap,
                "3d": self._create_3d_plot,
            }
            if chart_type in chart_creator:
                chart_creator[chart_type](
                    x_data, y_data, z_data, color_data, style, interactive
                )
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            # Save the visualization if a save path is provided
            if save_path:
                self._save_visualization(save_path)

            # Show the visualization
            plt.show()

        except Exception as e:
            self.logger.exception(f"An error occurred while visualizing data: {e}")
            advanced_logger.log(logging.ERROR, f"Data visualization failed: {e}")
            raise

    def _validate_data(
        self,
        x_axis: str,
        y_axis: str,
        z_axis: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        """
        Validate the input data columns for visualization.

        Args:
            x_axis (str): The data column to use for the x-axis.
            y_axis (str): The data column to use for the y-axis.
            z_axis (Optional[str]): The data column to use for the z-axis (for 3D plots).
            color (Optional[str]): The data column to use for color encoding.

        Returns:
            None

        Raises:
            ValueError: If any of the specified data columns are not found in the processed data.
        """
        missing_columns = [
            col
            for col in [x_axis, y_axis, z_axis, color]
            if col and col not in self.data
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in data: {', '.join(missing_columns)}"
            )

    def _validate_options(
        self, chart_type: str, style: Optional[str] = None, interactive: bool = True
    ) -> None:
        """
        Validate the chart options for visualization.

        Args:
            chart_type (str): The type of chart to create.
            style (Optional[str]): The style theme for the chart.
            interactive (bool): Whether to enable interactive plot controls.

        Returns:
            None

        Raises:
            ValueError: If any of the specified options are invalid.
        """
        valid_chart_types = ["line", "bar", "scatter", "heatmap", "3d"]
        if chart_type not in valid_chart_types:
            raise ValueError(
                f"Invalid chart type '{chart_type}'. Valid options are: {', '.join(valid_chart_types)}"
            )

        valid_styles = ["darkgrid", "whitegrid", "dark", "white", "ticks"]
        if style and style not in valid_styles:
            raise ValueError(
                f"Invalid style '{style}'. Valid options are: {', '.join(valid_styles)}"
            )

        if not isinstance(interactive, bool):
            raise ValueError(
                f"Invalid interactive value '{interactive}'. Must be a boolean."
            )

    def _prepare_data(
        self,
        x_axis: str,
        y_axis: str,
        z_axis: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Tuple[pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
        """
        Prepare the data for visualization by extracting the specified columns from the processed data.

        Args:
            x_axis (str): The data column to use for the x-axis.
            y_axis (str): The data column to use for the y-axis.
            z_axis (Optional[str]): The data column to use for the z-axis (for 3D plots).
            color (Optional[str]): The data column to use for color encoding.

        Returns:
            Tuple[pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]: The extracted data columns for visualization.
        """
        x_data = self.data[x_axis]
        y_data = self.data[y_axis]
        z_data = self.data[z_axis] if z_axis else None
        color_data = self.data[color] if color else None
        return x_data, y_data, z_data, color_data

    def _create_line_plot(
        self,
        x_data: pd.Series,
        y_data: pd.Series,
        z_data: Optional[pd.Series] = None,
        color_data: Optional[pd.Series] = None,
        style: Optional[str] = None,
        interactive: bool = True,
    ) -> None:
        """
        Create a line plot using the provided data and customization options.

        Args:
            x_data (pd.Series): The data for the x-axis.
            y_data (pd.Series): The data for the y-axis.
            color_data (Optional[pd.Series]): The data for color encoding.
            style (Optional[str]): The style theme for the chart.
            interactive (bool): Whether to enable interactive plot controls.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        if style:
            sns.set_style(style)
        sns.lineplot(x=x_data, y=y_data, hue=color_data)
        plt.title("Line Plot")
        plt.xlabel(x_data.name)
        plt.ylabel(y_data.name)
        if interactive:
            mplcursors.cursor(hover=True)

    def _create_bar_chart(
        self,
        x_data: pd.Series,
        y_data: pd.Series,
        z_data: Optional[pd.Series] = None,
        color_data: Optional[pd.Series] = None,
        style: Optional[str] = None,
        interactive: bool = True,
    ) -> None:
        """
        Create a bar chart using the provided data and customization options.

        Args:
            x_data (pd.Series): The data for the x-axis.
            y_data (pd.Series): The data for the y-axis.
            color_data (Optional[pd.Series]): The data for color encoding.
            style (Optional[str]): The style theme for the chart.
            interactive (bool): Whether to enable interactive plot controls.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        if style:
            sns.set_style(style)
        sns.barplot(x=x_data, y=y_data, hue=color_data)
        plt.title("Bar Chart")
        plt.xlabel(x_data.name)
        plt.ylabel(y_data.name)
        if interactive:
            mplcursors.cursor(hover=True)

    def _create_scatter_plot(
        self,
        x_data: pd.Series,
        y_data: pd.Series,
        z_data: Optional[pd.Series] = None,
        color_data: Optional[pd.Series] = None,
        style: Optional[str] = None,
        interactive: bool = True,
    ) -> None:
        """
        Create a scatter plot using the provided data and customization options.

        Args:
            x_data (pd.Series): The data for the x-axis.
            y_data (pd.Series): The data for the y-axis.
            color_data (Optional[pd.Series]): The data for color encoding.
            style (Optional[str]): The style theme for the chart.
            interactive (bool): Whether to enable interactive plot controls.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        if style:
            sns.set_style(style)
        sns.scatterplot(x=x_data, y=y_data, hue=color_data)
        plt.title("Scatter Plot")
        plt.xlabel(x_data.name)
        plt.ylabel(y_data.name)
        if interactive:
            mplcursors.cursor(hover=True)

    def _create_heatmap(
        self,
        x_data: pd.Series,
        y_data: pd.Series,
        z_data: Optional[pd.Series] = None,
        color_data: Optional[pd.Series] = None,
        style: Optional[str] = None,
        interactive: bool = True,
    ) -> None:
        """
        Create a heatmap using the provided data and customization options.

        Args:
            x_data (pd.Series): The data for the x-axis.
            y_data (pd.Series): The data for the y-axis.
            color_data (Optional[pd.Series]): The data for color encoding.
            style (Optional[str]): The style theme for the chart.
            interactive (bool): Whether to enable interactive plot controls.

        Returns:
            None
        """
        plt.figure(figsize=(10, 8))
        if style:
            sns.set_style(style)
        data_matrix = pd.pivot_table(
            pd.DataFrame({"x": x_data, "y": y_data, "color": color_data}),
            values="color",
            index="y",
            columns="x",
        )
        sns.heatmap(data_matrix, cmap="viridis")
        plt.title("Heatmap")
        if interactive:
            mplcursors.cursor(hover=True)

    def _create_3d_plot(
        self,
        x_data: pd.Series,
        y_data: pd.Series,
        z_data: pd.Series,
        color_data: Optional[pd.Series] = None,
        style: Optional[str] = None,
        interactive: bool = True,
    ) -> None:
        """
        Create a 3D plot using the provided data and customization options.

        Args:
            x_data (pd.Series): The data for the x-axis.
            y_data (pd.Series): The data for the y-axis.
            z_data (pd.Series): The data for the z-axis.
            color_data (Optional[pd.Series]): The data for color encoding.
            style (Optional[str]): The style theme for the chart.
            interactive (bool): Whether to enable interactive plot controls.

        Returns:
            None
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        if style:
            sns.set_style(style)
        ax.scatter(x_data, y_data, z_data, c=color_data, cmap="viridis")
        ax.set_title("3D Plot")
        ax.set_xlabel(x_data.name)
        ax.set_ylabel(y_data.name)
        ax.set_zlabel(z_data.name)
        if interactive:
            mplcursors.cursor(hover=True)

    def _save_visualization(self, save_path: str) -> None:
        """
        Save the current visualization to the specified file path.

        Args:
            save_path (str): The file path to save the visualization.

        Returns:
            None
        """
        try:
            plt.savefig(save_path)
            self.logger.info(f"Visualization saved to {save_path}")
            advanced_logger.log(logging.INFO, f"Visualization saved to {save_path}")
        except Exception as e:
            self.logger.exception(
                f"An error occurred while saving the visualization: {e}"
            )
            advanced_logger.log(logging.ERROR, f"Failed to save visualization: {e}")
            raise

    def visualize_metadata(
        self,
        metadata: Dict[str, Any],
        style: Optional[str] = None,
        interactive: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize the metadata using various chart types and customization options.

        Args:
            metadata (Dict[str, Any]): The metadata to visualize.
            style (Optional[str]): The style theme for the charts.
            interactive (bool): Whether to enable interactive plot controls.
            save_path (Optional[str]): The path to save the visualizations, if desired.

        Returns:
            None
        """
        try:
            # Validate input metadata
            self._validate_metadata(metadata)

            # Create visualizations for metadata
            self._visualize_metadata_counts(metadata, style, interactive, save_path)
            self._visualize_metadata_correlations(
                metadata, style, interactive, save_path
            )
            self._visualize_metadata_distributions(
                metadata, style, interactive, save_path
            )

            # Show all visualizations
            plt.show()

        except Exception as e:
            self.logger.exception(f"An error occurred while visualizing metadata: {e}")
            advanced_logger.log(logging.ERROR, f"Metadata visualization failed: {e}")
            raise

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Validate the input metadata to ensure it contains the required fields for visualization.

        Args:
            metadata (Dict[str, Any]): The metadata to validate.

        Returns:
            None

        Raises:
            ValueError: If any of the required metadata fields are missing.
        """
        required_fields = ["data_source_counts", "record_type_counts", "year_counts"]
        missing_fields = [field for field in required_fields if field not in metadata]
        if missing_fields:
            raise ValueError(
                f"Missing required metadata fields: {', '.join(missing_fields)}"
            )

    def _visualize_metadata_counts(
        self,
        metadata: Dict[str, Any],
        style: Optional[str] = None,
        interactive: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create visualizations for metadata counts.

        Args:
            metadata (Dict[str, Any]): The metadata to visualize.
            style (Optional[str]): The style theme for the charts.
            interactive (bool): Whether to enable interactive plot controls.
            save_path (Optional[str]): The path to save the visualizations, if desired.

        Returns:
            None
        """
        # Create a bar chart for the data source counts
        data_source_counts = pd.Series(metadata["data_source_counts"])
        plt.figure(figsize=(10, 6))
        sns.barplot(x=data_source_counts.index, y=data_source_counts, palette="viridis")
        plt.title("Data Source Counts")
        plt.xlabel("Data Source")
        plt.ylabel("Number of Records")
        if save_path:
            save_path_data_source_counts = f"{save_path}_data_source_counts.png"
            self._save_visualization(save_path_data_source_counts)

        # Create a pie chart for the distribution of record types
        record_type_counts = pd.Series(metadata["record_type_counts"])
        plt.figure(figsize=(10, 10))
        plt.pie(record_type_counts, labels=record_type_counts.index, autopct="%1.1f%%")
        plt.title("Distribution of Record Types")
        if interactive:
            mplcursors.cursor(hover=True)
        if save_path:
            save_path_record_type_counts = f"{save_path}_record_type_counts.png"
            self._save_visualization(save_path_record_type_counts)

        # Create a bar chart for the number of records per year
        year_counts = pd.Series(metadata["year_counts"]).sort_index()
        self._create_bar_chart(
            x_data=year_counts.index,
            y_data=year_counts,
            style=style,
            interactive=interactive,
        )
        plt.title("Number of Records per Year")
        plt.xlabel("Year")
        plt.ylabel("Number of Records")
        if save_path:
            save_path_year_counts = f"{save_path}_year_counts.png"
            self._save_visualization(save_path_year_counts)

    def _visualize_metadata_correlations(
        self,
        metadata: Dict[str, Any],
        style: Optional[str] = None,
        interactive: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create a heatmap for the correlation between numeric metadata fields.

        Args:
            metadata (Dict[str, Any]): The metadata to visualize.
            style (Optional[str]): The style theme for the charts.
            interactive (bool): Whether to enable interactive plot controls.
            save_path (Optional[str]): The path to save the visualizations, if desired.

        Returns:
            None
        """
        numeric_metadata = pd.DataFrame(
            {k: v for k, v in metadata.items() if isinstance(v, (int, float))}
        ).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_metadata, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap of Numeric Metadata Fields")
        if interactive:
            mplcursors.cursor(hover=True)
        if save_path:
            save_path_numeric_metadata_corr = f"{save_path}_numeric_metadata_corr.png"
            self._save_visualization(save_path_numeric_metadata_corr)

    def _visualize_metadata_distributions(
        self,
        metadata: Dict[str, Any],
        style: Optional[str] = None,
        interactive: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create a 3D scatter plot for the title similarity scores.

        Args:
            metadata (Dict[str, Any]): The metadata to visualize.
            style (Optional[str]): The style theme for the charts.
            interactive (bool): Whether to enable interactive plot controls.
            save_path (Optional[str]): The path to save the visualizations, if desired.

        Returns:
            None
        """
        if "title_similarity_scores" in metadata:
            title_similarity_scores = pd.DataFrame(metadata["title_similarity_scores"])
            self._create_3d_plot(
                x_data=title_similarity_scores["doc_id_1"],
                y_data=title_similarity_scores["doc_id_2"],
                z_data=title_similarity_scores["similarity_score"],
                color_data=None,
                style=style,
                interactive=interactive,
            )
            plt.title("Title Similarity Scores")
            plt.xlabel("Document ID 1")
            plt.ylabel("Document ID 2")
            plt.zlabel("Similarity Score")
            if save_path:
                save_path_title_similarity_scores = (
                    f"{save_path}_title_similarity_scores.png"
                )
                self._save_visualization(save_path_title_similarity_scores)

    def visualize_data(
        self,
        data: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        chart_type: str = "scatter",
        z_axis: Optional[str] = None,
        color: Optional[str] = None,
        style: Optional[str] = None,
        interactive: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize the processed data using the specified chart type and customization options.

        Args:
            data (pd.DataFrame): The processed data to visualize.
            x_axis (str): The column name to use for the x-axis.
            y_axis (str): The column name to use for the y-axis.
            chart_type (str): The type of chart to create ('scatter', 'line', 'bar', 'heatmap', '3d').
            z_axis (Optional[str]): The column name to use for the z-axis (for 3D plots).
            color (Optional[str]): The column name to use for color encoding.
            style (Optional[str]): The style theme for the chart.
            interactive (bool): Whether to enable interactive plot controls.
            save_path (Optional[str]): The path to save the visualization, if desired.

        Returns:
            None
        """
        try:
            # Validate input data and options
            self._validate_data(data, x_axis, y_axis, z_axis, color)
            self._validate_options(chart_type, style, interactive)

            # Prepare data for visualization
            x_data, y_data, z_data, color_data = self._prepare_data(
                data, x_axis, y_axis, z_axis, color
            )

            # Create the specified chart type
            chart_creator = {
                "scatter": self._create_scatter_plot,
                "line": self._create_line_plot,
                "bar": self._create_bar_chart,
                "heatmap": self._create_heatmap,
                "3d": self._create_3d_plot,
            }
            if chart_type in chart_creator:
                chart_creator[chart_type](
                    x_data, y_data, z_data, color_data, style, interactive
                )
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            # Add chart title and labels
            plt.title(f"{chart_type.capitalize()} Plot of {x_axis} vs {y_axis}")
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)

            # Save the visualization if a save path is provided
            if save_path:
                self._save_visualization(save_path)

            # Show the visualization
            plt.show()

        except Exception as e:
            self.logger.exception(f"An error occurred while visualizing data: {e}")
            advanced_logger.log(logging.ERROR, f"Data visualization failed: {e}")
            raise

    def visualize_data_flow(
        self,
        data_manager: UniversalDataManager,
        style: Optional[str] = None,
        interactive: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize the data flow through the UniversalDataManager using a directed graph.

        Args:
            data_manager (UniversalDataManager): The UniversalDataManager instance to visualize.
            style (Optional[str]): The style theme for the chart.
            interactive (bool): Whether to enable interactive plot controls.
            save_path (Optional[str]): The path to save the visualization, if desired.

        Returns:
            None
        """
        try:
            # Create a directed graph representing the data flow
            G = nx.DiGraph()

            # Add nodes for each data source
            for source in data_manager.data_sources:
                G.add_node(source, type="data_source")

            # Add nodes for each data processor
            for processor in data_manager.data_processors:
                G.add_node(processor.__class__.__name__, type="data_processor")

            # Add edges representing the data flow between sources and processors
            for source, processors in data_manager.source_processor_mapping.items():
                for processor in processors:
                    G.add_edge(source, processor.__class__.__name__)

            # Add edges representing the data flow between processors
            for i in range(len(data_manager.data_processors) - 1):
                G.add_edge(
                    data_manager.data_processors[i].__class__.__name__,
                    data_manager.data_processors[i + 1].__class__.__name__,
                )

            # Visualize the graph
            pos = nx.spring_layout(G)
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color="skyblue",
                edge_color="k",
                node_size=2000,
                font_size=10,
            )
            plt.title("Data Flow Visualization")
            if save_path:
                plt.savefig(save_path)
            plt.show()

        except Exception as e:
            self.logger.exception(f"An error occurred while visualizing data flow: {e}")
            advanced_logger.log(logging.ERROR, f"Data flow visualization failed: {e}")
            raise


def main() -> None:
    """
    Main function to demonstrate the advanced JSON processing capabilities with detailed logging and exception handling.
    Uses the ModernGUI class to provide a modern file browsing interface for selecting JSON files.
    Runs as a loop so that files can be processed repeatedly. Starting at the main menu, the user can choose to process a file, exit the program, or handle exceptions gracefully.
    """
    advanced_logger = AdvancedLoggingSystem()
    advanced_logger.setup_logging()
    advanced_logger.log(logging.INFO, "Logging system initialized and operational.")
    repository_path = "/home/lloyd/randomuselesscrap/codeanalytics/repository"
    gui = ModernGUI()
    data_manager = UniversalDataManager()
    model_initializer = ModelInitializer()
    model_initializer.download_model()
    model, tokenizer = model_initializer.load_model()
    data_visualizer = DataVisualizer({}, model_initializer, data_manager)

    try:
        while True:
            gui.run()
            if gui.file_path:
                json_processor = AdvancedJSONProcessor(gui.file_path, repository_path)
                processed_data = json_processor.process_data()
                data_visualizer.data = processed_data
                data_visualizer.visualize_data_flow(data_manager)
            else:
                break
    except Exception as e:
        advanced_logger.log(logging.ERROR, f"An unexpected error occurred: {e}")
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
    finally:
        gui.root.destroy()


if __name__ == "__main__":
    main()

# Path: codeanalytics/jsonprocessor.py
