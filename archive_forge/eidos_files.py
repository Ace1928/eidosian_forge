import os  # 📦 Provides functions for interacting with the operating system, such as file path manipulation.
import asyncio  # 🚀 Provides support for asynchronous programming, enabling concurrent operations.
import logging  # 🪵 Provides a flexible framework for emitting log messages from applications, crucial for debugging and monitoring.
from winsound import (
    Beep,
)  # 🎵 Provides access to basic sound-playing capabilities on Windows.
import psutil  # 📊 Provides functions for monitoring system resources, including CPU, memory, and disk usage.
import shutil  # 🗄️ Provides high-level file operations, such as copying and moving files and directories.
import json  # 📦 Provides functions for working with JSON data.
import pickle  # 💾 Provides functions for serializing and deserializing Python objects.
import hashlib  # 🔑 Provides functions for creating hash digests of data.
from collections import deque  # 🗄️ Provides a double-ended queue data structure.
from concurrent.futures import (
    ThreadPoolExecutor,
)  # 🧵 Provides tools for concurrent execution using threads.
from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)  # 🖋️ Provides type hinting for complex data structures, improving code clarity and enabling static analysis.
import pandas as pd  # 📊 Provides data manipulation and analysis tools.
from tqdm.auto import tqdm  # 📈 Provides progress bar functionality.
from nltk.tokenize import (
    word_tokenize,
)  # 📝 Provides functions for tokenizing text into words.
from transformers import (
    AutoTokenizer,
)  # 🤖 Provides tools for working with transformer models.
from llama_index.core import (
    Document,
)  # 🗂️ Provides a data structure for representing a document.
from llama_index.core.node_parser import (
    SentenceSplitter,
)  # ✂️ Provides tools for splitting text into sentences.
from llama_index.core.schema import (
    MetadataMode,
)  # 📚 Provides a class for collecting metadata from nodes using the LLM to extract metadata.
from llama_index.llms.huggingface import (
    HuggingFaceLLM,
)  # 🤖 Provides an interface for using Hugging Face models with LlamaIndex.
from pathlib import (
    Path,
)  # 📁 Provides a way to interact with files and directories in a more object-oriented way.
from typing import (
    Optional,
    Union,
)  # 🖋️ Provides type hinting for optional values and union types.

from eidos_config import (
    EidosConfig,
    DEFAULT_MODEL_NAME,
    MAX_TOKENS_PER_DOCUMENT,
    CHUNK_OVERLAP,
    DEFAULT_SENTENCE_CHUNK_SIZE,
    DEFAULT_SENTENCE_CHUNK_OVERLAP,
    DEFAULT_MAX_DOCUMENTS,
    DEFAULT_INITIAL_CHUNK_SIZE,
    DEFAULT_HIGH_RESOURCE_THRESHOLD,
)  # ⚙️ Imports the Eidos configuration class, containing default settings for the project.

logger = logging.getLogger(
    __name__
)  # 🪵 Initializes the logger for this module, allowing for structured logging.

class EidosDirectoryManager:
    """
    📁 Eidos Directory Management System.

    Manages the creation and organization of directories for the Eidos project.
    This class handles resource monitoring, adaptive chunking, and concurrent
    directory creation to ensure efficient and robust file system management.

    [all]
        __init__
        _initialize_directories
        _ensure_directory
        _start_disk_offloader
        _disk_offload_task
        _create_directory_on_disk
        _calculate_adaptive_chunk_size
        _create_all_directories
        _process_directories
        _log_chunk_info
        _create_directories_concurrently
        _create_single_directory
        get_directory_path
    """

    def __init__(self, config: EidosConfig):
        """
        Initializes the directory manager with configuration settings.

        Args:
            config (EidosConfig): An instance of EidosConfig containing the project's configuration.

        Initializes:
            self.config (EidosConfig): Stores the provided configuration.
            self.base_dir (str): The base directory for all Eidos files, defaulting to "/Development".
            self.high_resource_threshold (int): The resource usage threshold (in percentage) for triggering adaptive chunking, defaulting to 90.
            self.initial_chunk_size (int): The initial chunk size for directory processing, defaulting to 100.
            self.adaptive_chunking (bool): A flag to enable or disable adaptive chunking, defaulting to True.
            self.min_chunk_size (int): The minimum chunk size for adaptive chunking, defaulting to 1024.
            self.max_chunk_size (int): The maximum chunk size for adaptive chunking, defaulting to 10MB.
            self.disk_offload_delay (int): The delay (in seconds) between disk offload attempts, defaulting to 5.
            self.logger (logging.Logger): The logger for this class.
            self.executor (ThreadPoolExecutor): A thread pool executor for concurrent tasks.
            self._directory_paths (Dict[str, str]): A dictionary to store directory paths, keyed by their names.
            self._disk_offload_queue (asyncio.Queue): An asynchronous queue for offloading directory creation to disk.
            self._is_offloading (bool): A flag to indicate if disk offloading is active.

        Side Effects:
            Initializes all directories and starts the disk offloader task.
        """
        self.config = config  # ⚙️ Stores the provided configuration.
        self.base_dir = getattr(
            config, "base_dir", "/Development"
        )  # 📁 Sets the base directory, defaulting to "/Development".
        self.high_resource_threshold = getattr(
            config, "high_resource_threshold", 90
        )  # 🚦 Sets the high resource threshold, defaulting to 90%.
        self.initial_chunk_size = getattr(
            config, "initial_chunk_size", 100
        )  # 📦 Sets the initial chunk size, defaulting to 100.
        self.adaptive_chunking = getattr(
            config, "adaptive_chunking", True
        )  # ⚙️ Enables or disables adaptive chunking, defaulting to True.
        self.min_chunk_size = getattr(
            config, "min_chunk_size", 1024
        )  # 📏 Sets the minimum chunk size, defaulting to 1024.
        self.max_chunk_size = getattr(
            config, "max_chunk_size", 1024 * 1024 * 10
        )  # 📐 Sets the maximum chunk size, defaulting to 10MB.
        self.disk_offload_delay = getattr(
            config, "disk_offload_delay", 5
        )  # ⏳ Sets the disk offload delay, defaulting to 5 seconds.
        self.logger = logging.getLogger(
            __name__
        )  # 🪵 Initializes the logger for this class.
        self.executor = ThreadPoolExecutor()  # 🧵 Initializes a thread pool executor.
        self._directory_paths: Dict[str, str] = (
            {}
        )  # 🗺️ Initializes a dictionary to store directory paths.
        self._disk_offload_queue = (
            asyncio.Queue()
        )  # 🗄️ Initializes an asynchronous queue for disk offloading.
        self._is_offloading = False  # 🚦 Initializes the disk offloading flag to False.
        self.logger.debug(
            f"EidosDirectoryManager initialized with base_dir: {self.base_dir}"  # 🐛 Logs the initialization of the directory manager.
        )
        asyncio.run(
            self._initialize_directories()
        )  # 🚀 Runs the directory initialization asynchronously.
        asyncio.run(
            self._start_disk_offloader()
        )  # 🚀 Starts the disk offloader task asynchronously.

    async def _initialize_directories(self):
        """
        Initializes all directories.

        Calls the _create_all_directories method to create all necessary directories.

        Side Effects:
            Creates all directories defined in _create_all_directories.
        """
        await self._create_all_directories()  # 🚀 Asynchronously creates all directories.

    async def _ensure_directory(self, path: str) -> None:
        """
        Ensures a directory exists, creating it if necessary.

        Args:
            path (str): The path of the directory to ensure.

        Raises:
            OSError: If there is an error creating the directory.

        Side Effects:
            Creates the directory if it does not exist.
        """
        try:
            if not os.path.exists(path):  # 🔍 Checks if the directory exists.
                os.makedirs(
                    path, exist_ok=True
                )  # 🛠️ Creates the directory if it doesn't exist.
                self.logger.debug(
                    f"Created directory: {path}"
                )  # 🐛 Logs the creation of the directory.
        except OSError as e:  # 🚨 Catches any OSError during directory creation.
            self.logger.error(
                f"Error creating directory {path}: {e}"
            )  # 🐛 Logs the error.
            raise  # 💥 Re-raises the error.

    async def _start_disk_offloader(self):
        """
        Starts the disk offloading task.

        Checks if the disk offloader is already running and starts it if not.

        Side Effects:
            Starts the disk offload task if it is not already running.
        """
        if (
            not self._is_offloading
        ):  # 🔍 Checks if the disk offloader is not already running.
            self._is_offloading = True  # 🚦 Sets the disk offloading flag to True.
            asyncio.create_task(
                self._disk_offload_task()
            )  # 🚀 Creates and starts the disk offload task.

    async def _disk_offload_task(self):
        """
        Task to handle disk offloading.

        Continuously monitors the disk offload queue and creates directories on disk.

        Side Effects:
            Creates directories on disk from the offload queue.
        """
        while self._is_offloading:  # 🔄 Loops while disk offloading is active.
            try:
                path = (
                    await self._disk_offload_queue.get()
                )  # 🗄️ Gets a path from the disk offload queue.
                if path is None:  # 🔍 Checks if the path is None.
                    continue  # ⏭️ Skips to the next iteration if the path is None.
                self.logger.debug(
                    f"Offloading directory creation to disk: {path}"
                )  # 🐛 Logs the offloading of the directory creation.
                await asyncio.to_thread(
                    self._create_directory_on_disk, path
                )  # 🚀 Offloads the directory creation to a thread.
                self._disk_offload_queue.task_done()  # ✅ Signals that the task is done.
            except Exception as e:  # 🚨 Catches any exception during disk offload.
                self.logger.error(
                    f"Error during disk offload: {e}"
                )  # 🐛 Logs the error.
            await asyncio.sleep(
                self.disk_offload_delay
            )  # ⏳ Waits for the specified delay.

    def _create_directory_on_disk(self, path: str) -> None:
        """
        Creates a directory on disk.

        Args:
            path (str): The path of the directory to create.

        Raises:
            OSError: If there is an error creating the directory.

        Side Effects:
            Creates the directory on disk.
        """
        try:
            if not os.path.exists(path):  # 🔍 Checks if the directory exists.
                os.makedirs(
                    path, exist_ok=True
                )  # 🛠️ Creates the directory if it doesn't exist.
                self.logger.debug(
                    f"Created directory on disk: {path}"
                )  # 🐛 Logs the creation of the directory on disk.
        except OSError as e:  # 🚨 Catches any OSError during directory creation.
            self.logger.error(
                f"Error creating directory on disk {path}: {e}"
            )  # 🐛 Logs the error.
            raise  # 💥 Re-raises the error.

    def _calculate_adaptive_chunk_size(self) -> int:
        """
        Calculates the adaptive chunk size based on resource usage.

        Returns:
            int: The calculated chunk size.

        Logic:
            If adaptive chunking is disabled, returns the initial chunk size.
            Otherwise, calculates the CPU and memory usage and adjusts the chunk size accordingly.
            If resource usage is high, reduces the chunk size.
            If resource usage is low, increases the chunk size.
        """
        if not self.adaptive_chunking:  # 🔍 Checks if adaptive chunking is disabled.
            return self.initial_chunk_size  # 📦 Returns the initial chunk size.
        cpu_percent = psutil.cpu_percent()  # 📊 Gets the CPU usage percentage.
        mem_percent = (
            psutil.virtual_memory().percent
        )  # 🧠 Gets the memory usage percentage.
        resource_usage = max(
            cpu_percent, mem_percent
        )  # 📈 Gets the maximum of CPU and memory usage.
        if (
            resource_usage > self.high_resource_threshold
        ):  # 🚦 Checks if resource usage is above the threshold.
            chunk_size = max(
                self.min_chunk_size, self.initial_chunk_size // 2
            )  # 📏 Reduces the chunk size if resource usage is high.
            self.logger.debug(
                f"High resource usage, reducing chunk size to: {chunk_size}"  # 🐛 Logs the reduction in chunk size.
            )
        else:  # 📉 If resource usage is low.
            chunk_size = min(
                self.max_chunk_size, self.initial_chunk_size * 2
            )  # 📐 Increases the chunk size if resource usage is low.
            self.logger.debug(
                f"Low resource usage, increasing chunk size to: {chunk_size}"  # 🐛 Logs the increase in chunk size.
            )
        return chunk_size  # 📦 Returns the calculated chunk size.

    async def _create_all_directories(self) -> None:
        """
        Creates all Eidos directories.

        Defines a list of directories to create and their descriptions, then calls _process_directories to create them.

        Side Effects:
            Creates all directories defined in the directories_to_create list.
        """
        directories_to_create = [  # 📁 List of directories to create and their descriptions.
            (
                "saved_models",
                "💾 Saved models directory - Stores trained machine learning models.",
            ),
            (
                "datasets",
                "🗄️ Datasets directory - Contains various datasets used for training and analysis.",
            ),
            (
                "templates",
                "📝 Templates directory - Holds templates for documents, code, and other files.",
            ),
            ("tests", "🧪 Tests directory - Stores test suites and related files."),
            (
                "images",
                "🖼️ Images directory - Contains image files for various purposes.",
            ),
            ("audio", "🎧 Audio directory - Stores audio files."),
            ("video", "🎬 Video directory - Contains video files."),
            ("html", "🌐 HTML directory - Stores HTML files."),
            (
                "logs",
                "📜 Logs directory - Contains log files for debugging and monitoring.",
            ),
            (
                "notebooks",
                "📓 Notebooks directory - Stores Jupyter notebooks and similar files.",
            ),
            (
                "papers",
                "📄 Papers directory - Contains research papers and related documents.",
            ),
            (
                "environment",
                "⚙️ Environment directory - Stores environment configurations and setup files.",
            ),
            ("documents", "📑 Documents directory - Contains general documents."),
            (
                "python_repository",
                "🐍 Python repository directory - Stores Python scripts and modules.",
            ),
            ("csv", "📊 CSV directory - Contains CSV files."),
            ("compressed", "🗜️ Compressed directory - Stores compressed files."),
            (
                "applications",
                "🚀 Applications directory - Contains executable applications.",
            ),
            ("backups", "📦 Backups directory - Stores backup files."),
            (
                "chat_history_gpt",
                "💬 GPT chat history directory - Stores chat logs from GPT models.",
            ),
            (
                "chat_history_eidos",
                "🧠 Eidos chat history directory - Stores chat logs from Eidos.",
            ),
            (
                "sandbox",
                "🧰 Sandbox directory - A directory for testing and experimentation.",
            ),
            (
                "eidos",
                "👤 Personal Eidos directory - Stores personal Eidos-related files.",
            ),
            (
                "knowledge",
                "🧠 Eidos Knowledge Base directory - Root directory for all knowledge-related data.",
            ),
            (
                os.path.join("knowledge", "citeweb"),
                "🔗 Citations directory - Stores citation data.",
            ),
            (
                os.path.join("knowledge", "identity"),
                "🎭 Identity directory - Stores identity-related data.",
            ),
            (
                os.path.join("knowledge", "lessons"),
                "📚 Lessons learned directory - Stores lessons learned from projects.",
            ),
            (
                os.path.join("knowledge", "majormoments"),
                "📅 Major moments directory - Stores significant events and milestones.",
            ),
            (
                os.path.join("knowledge", "processed"),
                "🗂️ Processed data directory - Stores data that has been processed.",
            ),
            (
                os.path.join("knowledge", "raw"),
                "🕸️ Raw data directory - Stores unprocessed data.",
            ),
            (
                os.path.join("knowledge", "refined"),
                "✨ Refined data directory - Stores data that has been refined.",
            ),
            (
                os.path.join("knowledge", "speculative"),
                "❓ Speculative data directory - Stores speculative or experimental data.",
            ),
            (
                os.path.join("knowledge", "timeline"),
                "⏱️ Raw timeline data directory - Stores raw timeline data.",
            ),
            (
                os.path.join("knowledge", "verified"),
                "✅ Verified data directory - Stores verified data.",
            ),
            (
                os.path.join("knowledge", "diary"),
                "📝 Diary directory - Stores personal diary entries.",
            ),
        ]
        await self._process_directories(
            directories_to_create
        )  # 🚀 Asynchronously processes all directories.

    async def _process_directories(self, directories: List[Tuple[str, str]]) -> None:
        """
        Processes directories, creating them concurrently.

        Args:
            directories (List[Tuple[str, str]]): A list of tuples containing the subdirectory name and its description.

        Side Effects:
            Creates all directories provided in the input list.
        """
        for (
            subdir,
            description,
        ) in directories:  # 🔄 Loops through each directory and its description.
            await self._create_single_directory(
                subdir, description
            )  # 🚀 Asynchronously creates each directory.

    async def _log_chunk_info(
        self, start_index: int, chunk_size: int, total_directories: int
    ) -> None:
        """
        Logs information about the current chunk being processed.

        Args:
            start_index (int): The starting index of the current chunk.
            chunk_size (int): The size of the current chunk.
            total_directories (int): The total number of directories to process.

        Side Effects:
            Logs information about the current chunk being processed.
        """
        cpu_percent = psutil.cpu_percent()  # 📊 Gets the CPU usage percentage.
        mem_percent = (
            psutil.virtual_memory().percent
        )  # 🧠 Gets the memory usage percentage.
        chunk_number = (
            start_index // chunk_size
        ) + 1  # 🔢 Calculates the current chunk number.
        total_chunks = (
            total_directories - 1
        ) // chunk_size + 1  # 🔢 Calculates the total number of chunks.
        self.logger.info(
            f"Processing directory chunk {chunk_number}/{total_chunks} "  # 🐛 Logs the current chunk being processed.
            f"(CPU: {cpu_percent}%, Memory: {mem_percent}%)"  # 📊 Includes CPU and memory usage in the log.
        )

    async def _create_directories_concurrently(
        self, chunk: List[Tuple[str, str]]
    ) -> None:
        """
        Creates directories concurrently using a thread pool.

        Args:
            chunk (List[Tuple[str, str]]): A list of tuples containing the subdirectory name and its description for the current chunk.

        Side Effects:
            Creates all directories in the current chunk concurrently.
        """
        tasks = [  # 🧵 Creates a list of tasks for concurrent directory creation.
            self._create_single_directory(
                subdir, description
            )  # 🚀 Creates a task for each directory.
            for subdir, description in chunk  # 🔄 Loops through each directory in the chunk.
        ]
        await asyncio.gather(
            *tasks
        )  # 🚀 Asynchronously gathers and executes all tasks.

    async def _create_single_directory(self, subdir: str, description: str) -> None:
        """
        Creates a single directory and sets a dynamic attribute.

        Args:
            subdir (str): The subdirectory name.
            description (str): The description of the directory.

        Side Effects:
            Creates the directory and sets a dynamic attribute for its path.
        """
        path = os.path.join(
            self.base_dir, subdir
        )  # 📁 Creates the full path of the directory.
        try:
            if (
                psutil.disk_usage(self.base_dir).percent > self.high_resource_threshold
            ):  # 🚦 Checks if disk usage is above the threshold.
                await self._disk_offload_queue.put(
                    path
                )  # 🗄️ Queues the directory for disk offload.
                self.logger.debug(
                    f"Queued directory for disk offload: {path}"
                )  # 🐛 Logs that the directory is queued for offload.
            else:  # 📉 If disk usage is below the threshold.
                await self._ensure_directory(path)  # 🛠️ Ensures the directory exists.
                self.logger.debug(
                    f"{description}: {path}"
                )  # 🐛 Logs the creation of the directory.
            self._directory_paths[subdir.replace(os.sep, "_")] = (
                path  # 🗺️ Stores the directory path in the dictionary.
            )
        except Exception as e:  # 🚨 Catches any exception during directory creation.
            self.logger.error(
                f"Error creating directory {path}: {e}"
            )  # 🐛 Logs the error.

    def get_directory_path(self, subdir_key: str) -> str:
        """
        Retrieves the directory path by its key.

        Args:
            subdir_key (str): The key of the directory to retrieve.

        Returns:
            str: The path of the directory.

        Raises:
            ValueError: If the directory path is not found for the given key.
        """
        path = self._directory_paths.get(
            subdir_key
        )  # 🗺️ Gets the directory path from the dictionary.
        if not path:  # 🔍 Checks if the path is not found.
            self.logger.error(
                f"Directory path not found for key: {subdir_key}"
            )  # 🐛 Logs the error.
            raise ValueError(
                f"Directory path not found for key: {subdir_key}"
            )  # 💥 Raises a ValueError if the path is not found.
        return path  # 📦 Returns the directory path.


# Instantiate the directory manager to ensure directories are created on import
config = EidosConfig(
    base_dir="/Development"
)  # ⚙️ Creates an instance of EidosConfig with the base directory set to "/Development".
eidos_directories = EidosDirectoryManager(
    config
)  # 📁 Creates an instance of EidosDirectoryManager.


# Make individual directory paths easily accessible
BASE_DIR = eidos_directories.base_dir  # 📁 Gets the base directory.
SAVED_MODELS_DIR = eidos_directories.get_directory_path(
    "saved_models"
)  # 💾 Gets the path for the saved models directory.
DATASETS_DIR = eidos_directories.get_directory_path(
    "datasets"
)  # 🗄️ Gets the path for the datasets directory.
TEMPLATES_DIR = eidos_directories.get_directory_path(
    "templates"
)  # 📝 Gets the path for the templates directory.
TESTS_DIR = eidos_directories.get_directory_path(
    "tests"
)  # 🧪 Gets the path for the tests directory.
IMAGES_DIR = eidos_directories.get_directory_path(
    "images"
)  # 🖼️ Gets the path for the images directory.
AUDIO_DIR = eidos_directories.get_directory_path(
    "audio"
)  # 🎧 Gets the path for the audio directory.
VIDEO_DIR = eidos_directories.get_directory_path(
    "video"
)  # 🎬 Gets the path for the video directory.
HTML_DIR = eidos_directories.get_directory_path(
    "html"
)  # 🌐 Gets the path for the HTML directory.
LOGS_DIR = eidos_directories.get_directory_path(
    "logs"
)  # 📜 Gets the path for the logs directory.
NOTEBOOKS_DIR = eidos_directories.get_directory_path(
    "notebooks"
)  # 📓 Gets the path for the notebooks directory.
PAPERS_DIR = eidos_directories.get_directory_path(
    "papers"
)  # 📄 Gets the path for the papers directory.
ENVIRONMENT_DIR = eidos_directories.get_directory_path(
    "environment"
)  # ⚙️ Gets the path for the environment directory.
DOCUMENTS_DIR = eidos_directories.get_directory_path(
    "documents"
)  # 📑 Gets the path for the documents directory.
PYTHON_REPOSITORY_DIR = eidos_directories.get_directory_path(
    "python_repository"
)  # 🐍 Gets the path for the python repository directory.
CSV_DIR = eidos_directories.get_directory_path(
    "csv"
)  # 📊 Gets the path for the CSV directory.
COMPRESSED_DIR = eidos_directories.get_directory_path(
    "compressed"
)  # 🗜️ Gets the path for the compressed directory.
APPLICATIONS_DIR = eidos_directories.get_directory_path(
    "applications"
)  # 🚀 Gets the path for the applications directory.
BACKUPS_DIR = eidos_directories.get_directory_path(
    "backups"
)  # 📦 Gets the path for the backups directory.
CHAT_HISTORY_GPT_DIR = eidos_directories.get_directory_path(
    "chat_history_gpt"
)  # 💬 Gets the path for the GPT chat history directory.
CHAT_HISTORY_EIDOS_DIR = eidos_directories.get_directory_path(
    "chat_history_eidos"
)  # 🧠 Gets the path for the Eidos chat history directory.
SANDBOX_DIR = eidos_directories.get_directory_path(
    "sandbox"
)  # 🧰 Gets the path for the sandbox directory.
PERSONAL_DIR = eidos_directories.get_directory_path(
    "eidos"
)  # 👤 Gets the path for the personal Eidos directory.
KNOWLEDGE_BASE_DIR = eidos_directories.get_directory_path(
    "knowledge"
)  # 🧠 Gets the path for the knowledge base directory.
CITATIONS_DIR = eidos_directories.get_directory_path(
    "knowledge_citeweb"
)  # 🔗 Gets the path for the citations directory.
IDENTITY_DIR = eidos_directories.get_directory_path(
    "knowledge_identity"
)  # 🎭 Gets the path for the identity directory.
LESSONS_DIR = eidos_directories.get_directory_path(
    "knowledge_lessons"
)  # 📚 Gets the path for the lessons learned directory.
MAJOR_MOMENTS_DIR = eidos_directories.get_directory_path(
    "knowledge_majormoments"
)  # 📅 Gets the path for the major moments directory.
PROCESSED_DATA_DIR = eidos_directories.get_directory_path(
    "knowledge_processed"
)  # 🗂️ Gets the path for the processed data directory.
RAW_DATA_DIR = eidos_directories.get_directory_path(
    "knowledge_raw"
)  # 🕸️ Gets the path for the raw data directory.
REFINED_DATA_DIR = eidos_directories.get_directory_path(
    "knowledge_refined"
)  # ✨ Gets the path for the refined data directory.
SPECULATIVE_DATA_DIR = eidos_directories.get_directory_path(
    "knowledge_speculative"
)  # ❓ Gets the path for the speculative data directory.
RAW_TIMELINE_DIR = eidos_directories.get_directory_path(
    "knowledge_timeline"
)  # ⏱️ Gets the path for the raw timeline data directory.
VERIFIED_DATA_DIR = eidos_directories.get_directory_path(
    "knowledge_verified"
)  # ✅ Gets the path for the verified data directory.
DIARY_DIR = eidos_directories.get_directory_path(
    "knowledge_diary"
)  # 📝 Gets the path for the diary directory.


class TextProcessor:
    """
    ✨ TextProcessor: A versatile tool for handling text data. ✨

    This class provides functionalities for:
    - 📚 Loading documents from various sources (text, files, directories).
    - 🔤 Tokenizing text using an LLM tokenizer.
    - ✂️ Splitting text into sentences and chunks.
    - 👯‍♀️ Deduplicating documents.
    - 🧠 Managing memory efficiently during processing.
    - 🔄 Dynamically adjusting chunking parameters.
    - 💾 Offloading large data to disk when needed.
    - 🧹 Cleaning up temporary files.

    It's designed to be robust and flexible, capable of handling large datasets
    with ease. It integrates a chunkwise file reader for universal data handling.

    [all]
    Available interfaces:
        - __init__: Initializes the TextProcessor with configurations.
        - _load_llm_metadata_model: Loads the LLM tokenizer.
        - load_input: Loads input data from various sources.
        - read_files_chunkwise: Reads files chunkwise from a directory.
        - load_documents_from_directory: Loads documents from a directory.
        - load_document_from_file: Loads a single document from a file.
        - load_documents: Returns the currently loaded documents.
        - save_documents: Saves documents to a pickle file.
        - load_documents_from_pickle: Loads documents from a pickle file.
        - deduplicate_documents: Deduplicates documents based on content.
        - chunk_text: Chunks text based on token count.
        - split_large_document: Splits large documents exceeding token limits.
        - process_documents: Main pipeline for processing documents.
        - final_sentence_split: Performs final sentence-level chunking.
        - set_file_chunk_size: Dynamically sets the file chunk size.
        - set_memory_threshold: Dynamically sets the memory threshold.
        - set_sentence_chunk_size: Dynamically sets the sentence chunk size.
        - set_sentence_chunk_overlap: Dynamically sets the sentence chunk overlap.
        - set_chunk_size: Dynamically sets the chunk size.
        - cleanup_temporary_files: Cleans up temporary files.
    """

    def __init__(
        self,
        input_data: Optional[Union[List[Document], str, Path, dict]] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        max_documents: int = DEFAULT_MAX_DOCUMENTS,
        max_tokens_per_document: int = MAX_TOKENS_PER_DOCUMENT,
        sentence_chunk_size: int = DEFAULT_SENTENCE_CHUNK_SIZE,
        sentence_chunk_overlap: int = DEFAULT_SENTENCE_CHUNK_OVERLAP,
        eidos_directory_manager: Optional[EidosDirectoryManager] = None,
        file_chunk_size: int = DEFAULT_INITIAL_CHUNK_SIZE,
        memory_threshold: int = DEFAULT_HIGH_RESOURCE_THRESHOLD,
        llm: Optional[HuggingFaceLLM] = None,
        offload_to_disk: bool = False,
        temp_dir: Optional[str] = None,
    ):
        """
        🚀 Initializes the TextProcessor with configurations. 🚀

        This constructor sets up the TextProcessor with the provided parameters,
        including the input data, model name, chunking sizes, memory thresholds,
        and directory management. It also initializes the logger, tokenizer,
        and other internal variables.

        Args:
            input_data (Optional[Union[List[Document], str, Path, dict]]):
                Initial data to process. Can be a list of Documents, a string of text,
                a file path, or a directory path, or a dict with a 'text' key.
            model_name (str): Name of the pretrained model to use for tokenization.
            max_documents (int): Max number of documents to keep in memory.
            max_tokens_per_document (int): Token limit for splitting large docs.
            sentence_chunk_size (int): Sentence-based chunk size for the final split.
            sentence_chunk_overlap (int): Overlap tokens for the final sentence split.
            eidos_directory_manager (Optional[EidosDirectoryManager]): An instance of EidosDirectoryManager for file path management.
            file_chunk_size (int): Size (in bytes) used when reading files chunkwise.
            memory_threshold (int): Memory usage % at which to warn or adapt.
            llm (Optional[HuggingFaceLLM]): A custom HuggingFaceLLM instance if desired.
            offload_to_disk (bool): Whether to offload large data chunks to disk.
            temp_dir (Optional[str]): Directory to store offloaded chunks if enabled.

        Returns:
            None
        """
        self.logger = logging.getLogger(__name__)  # 🪵 Get the logger for this module.
        self.model_name = model_name  # 🏷️ Store the model name.
        self.max_documents = max_documents  # 📄 Store the max documents limit.
        self.max_tokens_per_document = (
            max_tokens_per_document  # 🔤 Store the max tokens per document.
        )
        self.sentence_chunk_size = (
            sentence_chunk_size  # ✂️ Store the sentence chunk size.
        )
        self.sentence_chunk_overlap = (
            sentence_chunk_overlap  # 겹 Store the sentence chunk overlap.
        )
        self.eidos_directory_manager = (
            eidos_directory_manager  # 🗂️ Store the directory manager.
        )
        self.file_chunk_size = file_chunk_size  # 💾 Store the file chunk size.
        self.memory_threshold = memory_threshold  # ⚠️ Store the memory threshold.
        self.documents: List[Document] = (
            []
        )  # 📜 Initialize an empty list to hold documents.
        self.llm_tokenizer = None  # 🤖 Initialize the LLM tokenizer to None.
        self._load_llm_metadata_model()  # ⚙️ Load the LLM tokenizer.
        self.tokenizer = (
            self.llm_tokenizer
        )  # 🔤 Set the tokenizer to the LLM tokenizer.
        self.offload_to_disk = offload_to_disk  # 💾 Store the offload to disk flag.
        self.temp_dir = (
            temp_dir if temp_dir else "temp_text_processor"
        )  # 📁 Set the temp directory.
        if self.offload_to_disk:  # 💾 If offloading to disk is enabled.
            os.makedirs(
                self.temp_dir, exist_ok=True
            )  # 📁 Create the temp directory if it doesn't exist.
        self.data_source = None  # 🗄️ Initialize the data source to None.
        self.chunks: List[str] = (
            []
        )  # 📦 Initialize an empty list to hold chunk file paths.
        self.llm = (
            llm if llm else HuggingFaceLLM(model_name=DEFAULT_MODEL_NAME)
        )  # 🤖 Initialize the LLM.

        if self.eidos_directory_manager:  # 🗂️ If a directory manager is provided.
            self.pickle_path = os.path.join(  # 📍 Construct the pickle path.
                self.eidos_directory_manager.get_directory_path("documents"),
                "documents.pkl",
            )
        else:  # 📍 If no directory manager is provided.
            self.pickle_path = "/content/extracted_text_output/documents.pkl"  # 📍 Set a default pickle path.
            self.logger.warning(  # ⚠️ Log a warning about using the default path.
                "EidosDirectoryManager not provided, using default pickle path."
            )

        if input_data is not None:  # 📥 If input data is provided.
            self.load_input(input_data)  # 📥 Load the input data.

    def _load_llm_metadata_model(self) -> None:
        """
        🤖 Loads the LLM tokenizer for metadata processing. 🤖

        This method attempts to load the tokenizer for the specified LLM model.
        If successful, it logs an info message. If it fails, it logs a warning
        and sets the tokenizer to None, indicating a fallback to NLTK.

        Args:
            None

        Returns:
            None
        """
        try:  # 🧪 Try to load the tokenizer.
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )  # 🤖 Load the tokenizer.
            self.logger.info(
                f"Loaded LLM tokenizer: {self.model_name}"
            )  # ℹ️ Log a success message.
        except Exception as e:  # ❌ If loading fails.
            self.llm_tokenizer = None  # 🤖 Set the tokenizer to None.
            self.logger.warning(  # ⚠️ Log a warning message.
                f"Could not load LLM tokenizer. Will fallback to NLTK word_tokenize. Error: {e}"
            )

    def load_input(self, input_data: Union[List[Document], str, Path, dict]) -> None:
        """
        📥 Loads and processes input data. 📥

        This method handles various types of input data:
        - A list of Document objects.
        - A string of text or a file/directory path.
        - A dictionary with a 'text' key.

        It extends the internal list of documents based on the input type.
        It also caps the number of documents to the `max_documents` limit.

        Args:
            input_data (Union[List[Document], str, Path, dict]):
                The input data to load.

        Returns:
            None
        """
        if isinstance(
            input_data, list
        ) and all(  # 📜 If the input is a list of Documents.
            isinstance(doc, Document) for doc in input_data
        ):
            self.documents.extend(input_data)  # 📜 Extend the document list.
        elif isinstance(input_data, str) or isinstance(
            input_data, Path
        ):  # 🛤️ If the input is a string or Path.
            path_str = str(input_data)  # 🛤️ Convert the path to a string.
            if os.path.isdir(path_str):  # 📁 If the path is a directory.
                self.load_documents_from_directory(
                    path_str
                )  # 📁 Load documents from the directory.
            elif os.path.isfile(path_str):  # 📄 If the path is a file.
                self.load_document_from_file(
                    path_str
                )  # 📄 Load the document from the file.
            else:  # 📝 If it's raw text.
                self.documents.append(  # 📝 Create a new Document object.
                    Document(text=path_str, metadata={"mode": MetadataMode.LLM})
                )
        elif (
            isinstance(input_data, dict) and "text" in input_data
        ):  # 🔑 If the input is a dict with a 'text' key.
            self.documents.append(  # 📝 Create a new Document object.
                Document(text=input_data["text"], metadata={"mode": MetadataMode.LLM})
            )
        else:  # ❌ If the input is invalid.
            raise ValueError(  # ❌ Raise a ValueError.
                "Input data must be a list of Documents, a text/path string, or a dict with 'text' key."
            )
        # Cap at max_documents
        self.documents = self.documents[
            : self.max_documents
        ]  # 📄 Limit the number of documents.

    ############################################################################
    # NEW: Full integration of read_files_chunkwise approach, as a method below #
    ############################################################################
    def read_files_chunkwise(
        self,
        directory: str,
        chunk_size: int = 1024 * 1024,
        memory_threshold: int = 90,
    ) -> str:
        """
        📖 Reads text and JSON files from a directory in memory-friendly chunks. 📖

        This method reads files from a directory in chunks, monitoring memory usage
        and logging warnings when usage is high. It supports both .txt and .json files.
        It merges seamlessly with the existing chunkwise approach for maximum
        flexibility and performance.

        Args:
            directory (str): The path to the directory containing files.
            chunk_size (int): The size of each chunk to read in bytes (default 1MB).
            memory_threshold (int): Memory usage % at which to log a warning (default 90).

        Returns:
            str: A single string containing the concatenated content of all valid files.
        """
        all_text = ""  # 📝 Initialize an empty string to hold all text.
        processed_files = set()  # 📁 Initialize an empty set to track processed files.
        file_queue = deque()  # 🗄️ Initialize a deque to hold file paths.
        if not os.path.exists(directory):  # 📁 If the directory does not exist.
            self.logger.error(
                f"Directory not found: {directory}"
            )  # ❌ Log an error message.
            return "Directory not found. Please check the directory path."  # ❌ Return an error message.

        # Gather target files
        for filename in os.listdir(
            directory
        ):  # 📁 Iterate through files in the directory.
            if filename.lower().endswith(
                (".txt", ".json")
            ):  # 📄 If the file is a .txt or .json file.
                file_queue.append(
                    os.path.join(directory, filename)
                )  # 🗄️ Add the file path to the queue.

        while file_queue:  # 🗄️ While there are files in the queue.
            file_path = file_queue.popleft()  # 🗄️ Get the next file path from the queue.
            if (
                file_path in processed_files
            ):  # 📁 If the file has already been processed.
                continue  # ⏭️ Skip to the next file.

            try:  # 🧪 Try to process the file.
                file_size = os.path.getsize(file_path)  # 📏 Get the file size.
                if file_size > 0:  # 📏 If the file size is greater than 0.
                    with open(
                        file_path, "r", encoding="utf-8"
                    ) as file:  # 📄 Open the file for reading.
                        if file_path.lower().endswith(
                            ".txt"
                        ):  # 📄 If the file is a .txt file.
                            # Read .txt in chunked manner
                            while True:  # 🔄 Read the file in chunks.
                                chunk = file.read(
                                    chunk_size
                                )  # 📖 Read a chunk of the file.
                                if not chunk:  # 📖 If the chunk is empty.
                                    break  # ⏭️ Break out of the loop.
                                all_text += chunk  # 📝 Add the chunk to the total text.
                        elif file_path.lower().endswith(
                            ".json"
                        ):  # 📄 If the file is a .json file.
                            # For .json, parse and then re-serialize to string
                            try:  # 🧪 Try to parse the JSON file.
                                json_data = json.load(file)  # 📦 Load the JSON data.
                                all_text += json.dumps(
                                    json_data
                                )  # 📝 Add the JSON data to the total text.
                            except (
                                json.JSONDecodeError
                            ) as e:  # ❌ If there is a JSON decode error.
                                self.logger.error(  # ❌ Log an error message.
                                    f"JSONDecodeError in {file_path}: {e}"
                                )
                                continue  # ⏭️ Skip to the next file.
                processed_files.add(
                    file_path
                )  # 📁 Add the file to the set of processed files.
                memory_usage = (
                    psutil.virtual_memory().percent
                )  # 🧠 Get the current memory usage.
                if (
                    memory_usage > memory_threshold
                ):  # 🧠 If the memory usage is above the threshold.
                    self.logger.warning(  # ⚠️ Log a warning message.
                        f"Memory usage high ({memory_usage}%), consider offloading to disk."
                    )
            except FileNotFoundError:  # ❌ If the file is not found.
                self.logger.error(
                    f"File not found: {file_path}"
                )  # ❌ Log an error message.
            except Exception as e:  # ❌ If there is an error reading the file.
                self.logger.error(
                    f"Error reading {file_path}: {e}"
                )  # ❌ Log an error message.

        return all_text  # 📝 Return the concatenated text.

    ############################################################################
    # Existing methods adapted to incorporate the new chunkwise approach above #
    ############################################################################
    def load_documents_from_directory(self, directory: str) -> None:
        """
        📁 Loads documents from .txt and .json files within the given directory. 📁

        This method uses the `read_files_chunkwise` method to load the content
        of all .txt and .json files in the specified directory. It then creates
        a single Document object containing the concatenated text of all files.

        Args:
            directory (str): The path to the directory containing files.

        Returns:
            None
        """
        self.logger.info(  # ℹ️ Log an info message.
            f"[INFO] Loading documents from directory (chunkwise): {directory}"
        )
        if not os.path.exists(directory):  # 📁 If the directory does not exist.
            self.logger.error(
                f"Directory not found: {directory}"
            )  # ❌ Log an error message.
            return  # ⏭️ Return early.

        # We can rely on the new chunkwise method to get the entire text, then store in a Document
        concatenated_text = (
            self.read_files_chunkwise(  # 📖 Read all files in the directory.
                directory,
                chunk_size=self.file_chunk_size,
                memory_threshold=self.memory_threshold,
            )
        )

        # Only create a document if there's some content
        if concatenated_text.strip():  # 📝 If there is content.
            # We can store the entire directory's content as a single Document
            self.documents.append(  # 📝 Create a new Document object.
                Document(
                    text=concatenated_text,
                    metadata={"mode": MetadataMode.LLM, "source_directory": directory},
                )
            )
        self.logger.info(  # ℹ️ Log an info message.
            f"[INFO] Finished loading chunkwise from directory: {directory}"
        )

    def load_document_from_file(self, file_path: str) -> None:
        """
        📄 Loads a single file (.txt or .json) chunkwise, storing it as a Document. 📄

        This method loads a single file using the `read_files_chunkwise` method
        by treating the single file as a 1-file directory. It creates a temporary
        directory, copies the file into it, reads the file, and then removes the
        temporary directory.

        Args:
            file_path (str): Full path to a .txt or .json file.

        Returns:
            None
        """
        self.logger.info(
            f"[INFO] Loading document from file: {file_path}"
        )  # ℹ️ Log an info message.
        if not os.path.exists(file_path):  # 📄 If the file does not exist.
            self.logger.error(
                f"File not found: {file_path}"
            )  # ❌ Log an error message.
            return  # ⏭️ Return early.

        # Reusing the chunkwise method by treating the single file as a 1-file directory
        dir_path, filename = os.path.split(
            file_path
        )  # 📁 Split the file path into directory and filename.
        tmp_dir = os.path.join(
            dir_path, "_TMP_SINGLE_FILE_"
        )  # 📁 Create a temporary directory path.

        try:  # 🧪 Try to process the file.
            os.makedirs(tmp_dir, exist_ok=True)  # 📁 Create the temporary directory.
            single_file_path = os.path.join(
                tmp_dir, filename
            )  # 📄 Create the path to the file in the temp directory.
            shutil.copy2(
                file_path, single_file_path
            )  # 📄 Copy the file to the temp directory.

            chunked_text = self.read_files_chunkwise(  # 📖 Read the file in chunks.
                tmp_dir,
                chunk_size=self.file_chunk_size,
                memory_threshold=self.memory_threshold,
            )

            shutil.rmtree(
                tmp_dir, ignore_errors=True
            )  # 📁 Remove the temporary directory.

            if chunked_text.strip():  # 📝 If there is content.
                self.documents.append(  # 📝 Create a new Document object.
                    Document(
                        text=chunked_text,
                        metadata={"mode": MetadataMode.LLM, "source_file": file_path},
                    )
                )
        except Exception as e:  # ❌ If there is an error processing the file.
            self.logger.error(
                f"Error processing file {file_path} chunkwise: {e}"
            )  # ❌ Log an error message.

    def load_documents(self) -> List[Document]:
        """
        📜 Returns the currently loaded list of documents. 📜

        This method simply returns the internal list of Document objects.

        Args:
            None

        Returns:
            List[Document]: The loaded Document objects.
        """
        return self.documents  # 📜 Return the list of documents.

    def save_documents(self, documents: List[Document]) -> None:
        """
        💾 Saves a list of Document objects to a pickle file. 💾

        This method saves the provided list of Document objects to a pickle file
        in the documents directory. The pickle path is determined by the
        `eidos_directory_manager` if available, otherwise a default path is used.

        Args:
            documents (List[Document]): The documents to be pickled.

        Returns:
            None
        """
        if self.eidos_directory_manager:  # 🗂️ If a directory manager is provided.
            self.pickle_path = os.path.join(  # 📍 Construct the pickle path.
                self.eidos_directory_manager.get_directory_path("documents"),
                "documents.pkl",
            )
        with open(self.pickle_path, "wb") as f:  # 💾 Open the pickle file for writing.
            pickle.dump(documents, f)  # 💾 Dump the documents to the pickle file.
        self.logger.info(  # ℹ️ Log an info message.
            f"[INFO] Saved {len(documents)} documents to {self.pickle_path}"
        )

    def load_documents_from_pickle(self) -> List[Document]:
        """
        📦 Loads Document objects from a pickle file, if present. 📦

        This method attempts to load Document objects from a pickle file.
        If the file exists and contains a list of Document objects, it returns
        the loaded documents. If the file does not exist or contains invalid data,
        it logs an error or warning and returns an empty list.

        Args:
            None

        Returns:
            List[Document]: The loaded Document objects (possibly empty).
        """
        self.logger.info(
            "[INFO] Attempting to load documents from pickle..."
        )  # ℹ️ Log an info message.
        if os.path.exists(self.pickle_path):  # 📦 If the pickle file exists.
            try:  # 🧪 Try to load the documents from the pickle file.
                with open(
                    self.pickle_path, "rb"
                ) as f:  # 📦 Open the pickle file for reading.
                    loaded_docs = pickle.load(
                        f
                    )  # 📦 Load the documents from the pickle file.
                if not isinstance(
                    loaded_docs, list
                ):  # 📦 If the loaded data is not a list.
                    raise ValueError(  # ❌ Raise a ValueError.
                        "Pickle file does not contain a list of Documents."
                    )
                self.logger.info(  # ℹ️ Log an info message.
                    f"[INFO] Loaded {len(loaded_docs)} documents from {self.pickle_path}"
                )
                return loaded_docs  # 📜 Return the loaded documents.
            except (
                Exception
            ) as e:  # ❌ If there is an error loading from the pickle file.
                self.logger.error(
                    f"[ERROR] Loading from pickle: {e}"
                )  # ❌ Log an error message.
                return []  # 📜 Return an empty list.
        else:  # 📦 If the pickle file does not exist.
            self.logger.warning(  # ⚠️ Log a warning message.
                f"[WARNING] Documents file not found at {self.pickle_path}"
            )
            return []  # 📜 Return an empty list.

    def deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
        👯‍♀️ Deduplicates documents by hashing their text content. 👯‍♀️

        This method deduplicates a list of Document objects by hashing their text
        content. It keeps only the first occurrence of each unique document based
        on the hash.

        Args:
            documents (List[Document]): Input Document list.

        Returns:
            List[Document]: Deduplicated Document list.
        """
        unique_hashes = set()  # 👯‍♀️ Initialize an empty set to hold unique hashes.
        deduped_docs = []  # 📜 Initialize an empty list to hold deduplicated documents.
        for doc in documents:  # 📜 Iterate through the documents.
            doc_hash = hashlib.md5(
                doc.text.encode("utf-8")
            ).hexdigest()  # 🔑 Generate a hash of the document text.
            if (
                doc_hash not in unique_hashes
            ):  # 🔑 If the hash is not in the set of unique hashes.
                unique_hashes.add(
                    doc_hash
                )  # 🔑 Add the hash to the set of unique hashes.
                deduped_docs.append(
                    doc
                )  # 📜 Add the document to the list of deduplicated documents.
        return deduped_docs  # 📜 Return the list of deduplicated documents.

    def chunk_text(self, text: str, max_tokens: int) -> List[str]:
        """
        ✂️ Splits text into smaller chunks based on a max token count. ✂️

        This method splits a given text into smaller chunks based on a maximum
        token count. It prefers to use the LLM tokenizer if available, and falls
        back to NLTK's word_tokenize if the LLM tokenizer is not available or fails.

        Args:
            text (str): Full input text.
            max_tokens (int): Token limit per chunk.

        Returns:
            List[str]: List of text chunks, each within the specified token limit.
        """
        if self.llm_tokenizer:  # 🤖 If the LLM tokenizer is available.
            try:  # 🧪 Try to use the LLM tokenizer.
                encoding = self.llm_tokenizer.encode(
                    text
                )  # 🤖 Encode the text using the LLM tokenizer.
                if (
                    len(encoding) <= max_tokens
                ):  # 🤖 If the encoded text is within the token limit.
                    return [text]  # 📝 Return the original text as a single chunk.
                chunks = []  # ✂️ Initialize an empty list to hold the chunks.
                for i in range(
                    0, len(encoding), max_tokens
                ):  # ✂️ Iterate through the encoded text in chunks.
                    subset = encoding[
                        i : i + max_tokens
                    ]  # ✂️ Get a subset of the encoded text.
                    subset_text = self.llm_tokenizer.decode(  # 🤖 Decode the subset of the encoded text.
                        subset, skip_special_tokens=True
                    )
                    chunks.append(
                        subset_text
                    )  # ✂️ Add the decoded subset to the list of chunks.
                return chunks  # ✂️ Return the list of chunks.
            except Exception as e:  # ❌ If there is an error using the LLM tokenizer.
                self.logger.warning(  # ⚠️ Log a warning message.
                    f"LLM tokenizer chunking failed; fallback to word_tokenize. Error: {e}"
                )

        # Fallback to NLTK
        tokens = word_tokenize(text)  # 🔤 Tokenize the text using NLTK.
        if (
            len(tokens) <= max_tokens
        ):  # 🔤 If the number of tokens is within the token limit.
            return [text]  # 📝 Return the original text as a single chunk.
        chunks = []  # ✂️ Initialize an empty list to hold the chunks.
        for i in range(
            0, len(tokens), max_tokens
        ):  # ✂️ Iterate through the tokens in chunks.
            chunk_tokens = tokens[i : i + max_tokens]  # ✂️ Get a subset of the tokens.
            chunk = " ".join(chunk_tokens)  # ✂️ Join the tokens into a string.
            chunks.append(chunk)  # ✂️ Add the chunk to the list of chunks.
        return chunks  # ✂️ Return the list of chunks.

    def split_large_document(self, doc: Document) -> List[Document]:
        """
        ✂️ Splits a document if it exceeds the token limit. ✂️

        This method checks if a document exceeds the maximum token limit.
        If it does, it splits the document into smaller sub-documents using
        the SentenceSplitter.

        Args:
            doc (Document): Document to check and potentially split.

        Returns:
            List[Document]: 1..N sub-documents.
        """
        token_count = len(
            doc.text.split()
        )  # 🔤 Get the number of tokens in the document.
        if (
            token_count <= self.max_tokens_per_document
        ):  # 🔤 If the token count is within the limit.
            return [doc]  # 📝 Return the original document as a single document.
        self.logger.info(  # ℹ️ Log an info message.
            f"[INFO] Document exceeds token limit ({token_count} tokens). Splitting..."
        )

        splitter = SentenceSplitter(  # ✂️ Initialize a SentenceSplitter.
            chunk_size=self.max_tokens_per_document,
            chunk_overlap=self.sentence_chunk_overlap,
        )
        split_nodes = splitter.get_nodes_from_documents(
            [doc]
        )  # ✂️ Split the document into nodes.
        new_docs = []  # 📜 Initialize an empty list to hold the new documents.
        for node in split_nodes:  # ✂️ Iterate through the nodes.
            new_docs.append(  # 📝 Create a new Document object.
                Document(
                    text=node.get_content(metadata_mode=MetadataMode.LLM),
                    metadata={"mode": MetadataMode.LLM},
                )
            )
        return new_docs  # 📜 Return the list of new documents.

    def process_documents(
        self,
        existing_documents: List[Document],
        news: Optional[pd.DataFrame] = None,
    ) -> List[Document]:
        """
        ⚙️ Main pipeline for processing documents. ⚙️

        This method orchestrates the core document processing steps, including merging,
        splitting, and deduplication. It takes existing documents and optionally
        new documents from a DataFrame, processes them, and returns a list of
        processed documents.

        [all]
            Merges existing documents with optional new documents from a DataFrame.
            Splits large documents that exceed the token limit.
            Deduplicates the documents.
            Returns the final list of processed documents, capped to `max_documents`.

        Args:
            existing_documents (List[Document]): A list of pre-loaded `Document` objects.
                These are the documents that have already been loaded into the system.
            news (Optional[pd.DataFrame], optional): An optional pandas DataFrame containing
                new documents to be added. It should have 'title' and 'text' columns.
                Defaults to None.

        Returns:
            List[Document]: A list of processed `Document` objects, which have been
                merged, split, and deduplicated.

        Raises:
            Exception: If any error occurs during the document processing pipeline.

        Side Effects:
            Logs information about the document processing steps.
        """
        documents = existing_documents[
            :
        ]  # 📜 Create a copy of the existing documents to avoid modifying the original list.
        # Merge from DataFrame
        if (
            news is not None and not news.empty
        ):  # 📰 Check if a DataFrame of news is provided and is not empty.
            self.logger.info(
                "[INFO] Creating news documents from DataFrame..."
            )  # ℹ️ Log an info message indicating that news documents are being created from the DataFrame.
            for (
                i,
                row,
            ) in tqdm(  # 📰 Iterate through the rows of the DataFrame using tqdm for progress tracking.
                news.iterrows(), total=len(news), desc="Creating news docs"
            ):
                text_content = f"{row['title']}: {row['text']}"  # 📰 Create the text content by combining the title and text from the DataFrame row.
                documents.append(  # 📝 Create a new Document object and append it to the list of documents.
                    Document(text=text_content, metadata={"mode": MetadataMode.LLM})
                )

        # Limit doc count
        documents = documents[
            : self.max_documents
        ]  # 📄 Limit the number of documents to the maximum allowed, as defined by `self.max_documents`.

        # Split large docs
        split_documents = []  # ✂️ Initialize an empty list to hold the split documents.
        for doc in documents:  # 📜 Iterate through the documents.
            split_documents.extend(
                self.split_large_document(doc)
            )  # ✂️ Split the document if it exceeds the token limit, extending the `split_documents` list with the results.

        # Deduplicate
        deduped = self.deduplicate_documents(
            split_documents
        )  # 👯‍♀️ Deduplicate the documents using the `deduplicate_documents` method.
        return deduped  # 🚀 Return the final list of processed, deduplicated documents.

    def final_sentence_split(self, documents: List[Document]) -> List[Document]:
        """
        Performs a final sentence-level chunking pass on all documents.

        This method takes a list of documents and splits them into smaller chunks
        based on sentences, using the `SentenceSplitter`. This is typically the
        last step in the document processing pipeline before the documents are
        used for further processing.

        [all]
            Splits documents into sentence-level chunks.
            Handles cases where no documents are provided.
            Logs information about the splitting process.

        Args:
            documents (List[Document]): A list of `Document` objects to be chunked.

        Returns:
            List[Document]: A list of `Document` objects, where each document
                represents a sentence-level chunk.

        Raises:
            Exception: If any error occurs during the sentence splitting process.

        Side Effects:
            Logs information about the sentence splitting process.
        """
        if not documents:  # 🧐 Check if the input list of documents is empty.
            self.logger.error(
                "[ERROR] No documents available to process."
            )  # 🚨 Log an error message if no documents are available.
            return []  # ↩️ Return an empty list if no documents are provided.

        splitter = SentenceSplitter(  # ✂️ Initialize a SentenceSplitter with the configured chunk size and overlap.
            chunk_size=self.sentence_chunk_size,
            chunk_overlap=self.sentence_chunk_overlap,
        )
        self.logger.info(
            "[INFO] Performing final sentence splitting..."
        )  # ℹ️ Log an info message indicating that sentence splitting is starting.

        nodes = list(  # ✂️ Split the documents into nodes using the SentenceSplitter and convert the generator to a list.
            tqdm(
                splitter.get_nodes_from_documents(documents),
                total=len(documents),
                desc="Generating nodes",
            )
        )
        new_docs = []  # 📜 Initialize an empty list to hold the new documents.
        for node in nodes:  # ✂️ Iterate through the nodes.
            new_docs.append(  # 📝 Create a new Document object from each node and append it to the list of new documents.
                Document(
                    text=node.get_content(metadata_mode=MetadataMode.LLM),
                    metadata={"mode": MetadataMode.LLM},
                )
            )
        self.logger.info(
            f"[INFO] Generated {len(new_docs)} nodes."
        )  # ℹ️ Log an info message indicating the number of nodes generated.
        return new_docs  # 🚀 Return the list of new documents, each representing a sentence-level chunk.

    ############################################################################
    # Dynamic setters and cleanup routines
    ############################################################################
    def set_file_chunk_size(self, new_chunk_size: int) -> None:
        """Dynamically adjusts the file reading chunk size for large files.

        This method allows for the dynamic adjustment of the chunk size used when
        reading large files. This can be useful for optimizing memory usage and
        processing speed based on the size of the files being processed.

        [all]
            Dynamically adjusts the file chunk size.
            Logs the change in chunk size.

        Args:
            new_chunk_size (int): The new chunk size to be set.

        Returns:
            None

        Raises:
            None

        Side Effects:
            Updates the `file_chunk_size` attribute.
            Logs the change in chunk size.
        """
        if (
            new_chunk_size > 0 and new_chunk_size != self.file_chunk_size
        ):  # 🧐 Check if the new chunk size is valid and different from the current one.
            self.logger.info(
                f"Updating file chunk size from {self.file_chunk_size} to {new_chunk_size}"
            )  # ℹ️ Log an info message indicating the change in chunk size.
            self.file_chunk_size = (
                new_chunk_size  # ⚙️ Update the file chunk size with the new value.
            )

    def set_memory_threshold(self, new_threshold: int) -> None:
        """Dynamically adjusts the memory usage threshold for chunk reading.

        This method allows for the dynamic adjustment of the memory usage threshold
        used for chunk reading. This can be useful for optimizing memory usage and
        processing speed based on the available system resources.

        [all]
            Dynamically adjusts the memory threshold.
            Logs the change in memory threshold.

        Args:
            new_threshold (int): The new memory threshold to be set, as a percentage (0-100).

        Returns:
            None

        Raises:
            None

        Side Effects:
            Updates the `memory_threshold` attribute.
            Logs the change in memory threshold.
        """
        if (
            0 < new_threshold <= 100 and new_threshold != self.memory_threshold
        ):  # 🧐 Check if the new threshold is valid and different from the current one.
            self.logger.info(
                f"Updating memory threshold from {self.memory_threshold} to {new_threshold}%"
            )  # ℹ️ Log an info message indicating the change in memory threshold.
            self.memory_threshold = (
                new_threshold  # ⚙️ Update the memory threshold with the new value.
            )

    def set_sentence_chunk_size(self, new_chunk_size: int) -> None:
        """Dynamically adjusts the sentence chunk size for final splitting.

        This method allows for the dynamic adjustment of the chunk size used when
        splitting sentences. This can be useful for optimizing the granularity of
        the final sentence-level chunks.

        [all]
            Dynamically adjusts the sentence chunk size.
            Logs the change in sentence chunk size.

        Args:
            new_chunk_size (int): The new sentence chunk size to be set.

        Returns:
            None

        Raises:
            None

        Side Effects:
            Updates the `sentence_chunk_size` attribute.
            Logs the change in sentence chunk size.
        """
        if (
            new_chunk_size > 0 and new_chunk_size != self.sentence_chunk_size
        ):  # 🧐 Check if the new chunk size is valid and different from the current one.
            self.logger.info(
                f"Updating sentence chunk size from {self.sentence_chunk_size} to {new_chunk_size}"
            )  # ℹ️ Log an info message indicating the change in sentence chunk size.
            self.sentence_chunk_size = (
                new_chunk_size  # ⚙️ Update the sentence chunk size with the new value.
            )

    def set_sentence_chunk_overlap(self, new_overlap: int) -> None:
        """Dynamically adjusts the sentence chunk overlap for final splitting.

        This method allows for the dynamic adjustment of the overlap between
        sentence chunks. This can be useful for ensuring that context is not lost
        when splitting sentences.

        [all]
            Dynamically adjusts the sentence chunk overlap.
            Logs the change in sentence chunk overlap.

        Args:
            new_overlap (int): The new sentence chunk overlap to be set.

        Returns:
            None

        Raises:
            None

        Side Effects:
            Updates the `sentence_chunk_overlap` attribute.
            Logs the change in sentence chunk overlap.
        """
        if (
            new_overlap >= 0 and new_overlap != self.sentence_chunk_overlap
        ):  # 🧐 Check if the new overlap is valid and different from the current one.
            self.logger.info(
                f"Updating sentence chunk overlap from {self.sentence_chunk_overlap} to {new_overlap}"
            )  # ℹ️ Log an info message indicating the change in sentence chunk overlap.
            self.sentence_chunk_overlap = (
                new_overlap  # ⚙️ Update the sentence chunk overlap with the new value.
            )

    def set_chunk_size(self, new_chunk_size: int) -> None:
        """
        Dynamically adjusts the chunk size for offloading or reading.
        Could trigger re-chunking or reloading if needed.

        This method allows for the dynamic adjustment of the chunk size used for
        offloading or reading data. This can be useful for optimizing memory usage
        and processing speed based on the available system resources and the size
        of the data being processed.

        [all]
            Dynamically adjusts the chunk size.
            Logs the change in chunk size.
            Potentially triggers re-chunking or reloading of data.

        Args:
            new_chunk_size (int): The new chunk size to be set.

        Returns:
            None

        Raises:
            None

        Side Effects:
            Updates the `file_chunk_size` attribute.
            Logs the change in chunk size.
            May trigger re-chunking or reloading of data if needed.
        """
        if (
            new_chunk_size > 0 and new_chunk_size != self.file_chunk_size
        ):  # 🧐 Check if the new chunk size is valid and different from the current one.
            self.logger.info(
                f"Updating processing chunk size from {self.file_chunk_size} to {new_chunk_size}"
            )  # ℹ️ Log an info message indicating the change in chunk size.
            self.file_chunk_size = (
                new_chunk_size  # ⚙️ Update the file chunk size with the new value.
            )
            # If data is offloaded or loaded, re-chunking logic could be triggered here. # ⚙️ A comment indicating that re-chunking logic could be triggered here.

    def cleanup_temporary_files(self) -> None:
        """Removes temporary files if offload_to_disk is enabled.

        This method cleans up temporary files and directories created during the
        offloading process. It is called when the processing is complete or when
        the temporary files are no longer needed.

        [all]
            Removes temporary files and directories.
            Handles cases where the temporary directory does not exist.
            Logs information about the cleanup process.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: If any error occurs during the cleanup process.

        Side Effects:
            Removes the temporary directory and its contents.
            Creates a new empty temporary directory for future use.
            Logs information about the cleanup process.
        """
        if self.offload_to_disk and os.path.exists(
            self.temp_dir
        ):  # 🧐 Check if offloading to disk is enabled and the temporary directory exists.
            self.logger.info(
                f"Cleaning up temporary files in {self.temp_dir}"
            )  # ℹ️ Log an info message indicating that temporary files are being cleaned up.
            try:  # ⛑️ Use a try-except block to handle potential errors during cleanup.
                shutil.rmtree(
                    self.temp_dir
                )  # 🗑️ Remove the temporary directory and its contents.
                os.makedirs(
                    self.temp_dir, exist_ok=True
                )  # 📁 Create a new empty temporary directory for future use.
            except Exception as e:  # 🚨 Catch any exceptions that occur during cleanup.
                self.logger.error(
                    f"Error cleaning up temp directory: {e}"
                )  # 🚨 Log an error message if an exception occurs during cleanup.

    def safe_cleanup(
        self,
        protected_paths: Optional[List[str]] = None,
        force: bool = False,
    ):
        """
        Safely cleans up the base directory if it's empty, with optional protection for specified paths.

        This method provides a safe way to clean up the base directory, ensuring that
        protected paths are not removed and that the directory is only removed if it
        is empty or if the `force` flag is set.

        [all]
            Safely cleans up the base directory.
            Protects specified paths from being cleaned up.
            Handles cases where the directory does not exist or is not empty.
            Logs information about the cleanup process.

        Args:
            protected_paths (Optional[List[str]], optional): A list of paths that should not be cleaned up.
                Defaults to None, which uses a default list of protected paths.
            force (bool, optional): If True, attempts to remove the directory even if it's not empty.
                Defaults to False.

        Returns:
            None

        Raises:
            Exception: If any error occurs during the cleanup process.

        Side Effects:
            Removes the base directory if it is empty or if the `force` flag is set.
            Logs information about the cleanup process.
        """
        if (
            not self.eidos_directory_manager
        ):  # 🧐 Check if the EidosDirectoryManager is available.
            self.logger.warning(
                "EidosDirectoryManager not available, skipping safe cleanup."
            )  # ⚠️ Log a warning message if the EidosDirectoryManager is not available.
            print(
                "EidosDirectoryManager not available, skipping safe cleanup."
            )  # ⚠️ Print a warning message to the console if the EidosDirectoryManager is not available.
            return  # ↩️ Return early if the EidosDirectoryManager is not available.

        default_protected_paths = [
            "/Development"
        ]  # 🛡️ Define a default list of protected paths.
        if (
            protected_paths is None
        ):  # 🧐 Check if a list of protected paths was provided.
            protected_paths = default_protected_paths  # 🛡️ Use the default list of protected paths if none was provided.
        else:  # 🛡️ If a list of protected paths was provided.
            protected_paths = list(
                set(default_protected_paths + protected_paths)
            )  # 🛡️ Combine the default and provided protected paths, removing duplicates.

        try:  # ⛑️ Use a try-except block to handle potential errors during cleanup.
            base_dir = (
                self.eidos_directory_manager.base_dir
            )  # 📁 Get the base directory from the EidosDirectoryManager.
            if (
                base_dir in protected_paths
            ):  # 🛡️ Check if the base directory is in the list of protected paths.
                self.logger.warning(
                    f"Attempted to clean up protected directory {base_dir}, which is prohibited. Skipping cleanup."
                )  # ⚠️ Log a warning message if an attempt is made to clean up a protected directory.
                print(
                    f"Attempted to clean up protected directory {base_dir}, which is prohibited. Skipping cleanup."
                )  # ⚠️ Print a warning message to the console if an attempt is made to clean up a protected directory.
                return  # ↩️ Return early if the base directory is protected.

            if not os.path.exists(base_dir):  # 🧐 Check if the base directory exists.
                self.logger.info(
                    f"Directory does not exist, skipping cleanup: {base_dir}"
                )  # ℹ️ Log an info message if the base directory does not exist.
                print(
                    f"Directory does not exist, skipping cleanup: {base_dir}"
                )  # ℹ️ Print an info message to the console if the base directory does not exist.
                return  # ↩️ Return early if the base directory does not exist.

            if not os.path.isdir(
                base_dir
            ):  # 🧐 Check if the base directory is a directory.
                self.logger.warning(
                    f"Path {base_dir} exists but is not a directory. Skipping cleanup."
                )  # ⚠️ Log a warning message if the base directory is not a directory.
                print(
                    f"Path {base_dir} exists but is not a directory. Skipping cleanup."
                )  # ⚠️ Print a warning message to the console if the base directory is not a directory.
                return  # ↩️ Return early if the base directory is not a directory.

            if not force and os.listdir(
                base_dir
            ):  # 🧐 Check if the base directory is empty and if force is not set.
                self.logger.info(
                    f"Directory is not empty, skipping cleanup: {base_dir}"
                )  # ℹ️ Log an info message if the base directory is not empty.
                print(
                    f"Directory is not empty, skipping cleanup: {base_dir}"
                )  # ℹ️ Print an info message to the console if the base directory is not empty.
                return  # ↩️ Return early if the base directory is not empty and force is not set.

            try:  # ⛑️ Use a try-except block to handle potential errors during directory removal.
                if force:  # 🧐 Check if the force flag is set.
                    shutil.rmtree(
                        base_dir
                    )  # 🗑️ Remove the base directory and its contents forcefully.
                    self.logger.info(
                        f"Forcefully cleaned up directory: {base_dir}"
                    )  # ℹ️ Log an info message indicating that the directory was forcefully cleaned up.
                    print(
                        f"Forcefully cleaned up directory: {base_dir}"
                    )  # ℹ️ Print an info message to the console indicating that the directory was forcefully cleaned up.
                else:  # 🗑️ If the force flag is not set.
                    os.rmdir(base_dir)  # 🗑️ Remove the base directory if it is empty.
                    self.logger.info(
                        f"Cleaned up empty directory: {base_dir}"
                    )  # ℹ️ Log an info message indicating that the empty directory was cleaned up.
                    print(
                        f"Cleaned up empty directory: {base_dir}"
                    )  # ℹ️ Print an info message to the console indicating that the empty directory was cleaned up.
            except (
                OSError
            ) as e:  # 🚨 Catch any OSError exceptions that occur during directory removal.
                self.logger.error(
                    f"Could not remove directory {base_dir}: {e}"
                )  # 🚨 Log an error message if the directory could not be removed.
                print(
                    f"Could not remove directory {base_dir}: {e}"
                )  # 🚨 Print an error message to the console if the directory could not be removed.

        except (
            Exception
        ) as e:  # 🚨 Catch any other exceptions that occur during cleanup.
            self.logger.error(
                f"Could not clean up directory: {e}"
            )  # 🚨 Log an error message if the directory could not be cleaned up.
            print(
                f"Could not clean up directory: {e}"
            )  # 🚨 Print an error message to the console if the directory could not be cleaned up.

    def cleanup(self) -> None:
        """
        Removes temporary files and directories used for offloading or chunking,
        and attempts a safe cleanup of the base directory.

        This method orchestrates the cleanup of temporary files and directories
        created during the offloading or chunking process, and also attempts a
        safe cleanup of the base directory.

        [all]
            Removes temporary files and directories.
            Attempts a safe cleanup of the base directory.
            Handles cases where the temporary directory does not exist.
            Logs information about the cleanup process.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: If any error occurs during the cleanup process.

        Side Effects:
            Removes the temporary directory and its contents.
            Creates a new empty temporary directory for future use.
            Attempts a safe cleanup of the base directory.
            Logs information about the cleanup process.
        """
        if self.offload_to_disk and os.path.exists(
            self.temp_dir
        ):  # 🧐 Check if offloading to disk is enabled and the temporary directory exists.
            self.logger.info(
                f"Cleaning up temp files in {self.temp_dir}"
            )  # ℹ️ Log an info message indicating that temporary files are being cleaned up.
            try:  # ⛑️ Use a try-except block to handle potential errors during cleanup.
                shutil.rmtree(
                    self.temp_dir
                )  # 🗑️ Remove the temporary directory and its contents.
                os.makedirs(
                    self.temp_dir, exist_ok=True
                )  # 📁 Create a new empty temporary directory for future use.
                self.chunks.clear()  # 🧹 Clear the list of chunks.
            except Exception as e:  # 🚨 Catch any exceptions that occur during cleanup.
                self.logger.error(
                    f"Error cleaning up temporary directory {self.temp_dir}: {e}"
                )  # 🚨 Log an error message if an exception occurs during cleanup.

        # Attempt safe cleanup of the base directory
        if (
            self.eidos_directory_manager
        ):  # 🧐 Check if the EidosDirectoryManager is available.
            self.safe_cleanup()  # 🧹 Attempt a safe cleanup of the base directory.

    ############################################################################
    # 💾 Offloading Logic (Optional, for Extremely Large Text Sources) 💾
    ############################################################################
    def _load_chunk(self, chunk_index: int) -> str:
        """
        📚 Loads a chunk of text from disk. 📚

        This method is used to load a specific chunk of text from a file on disk,
        typically when dealing with large text sources that have been split into
        smaller, manageable chunks. This is part of the offloading mechanism to
        handle large datasets that may not fit into memory.

        [all]
            Loads a chunk of text from disk.
            Handles file not found errors.
            Handles general exceptions during file loading.

        Args:
            chunk_index (int): The index of the chunk to load. This corresponds to the
                position of the chunk's file path in the `self.chunks` list.

        Returns:
            str: The text content of the loaded chunk.

        Raises:
            FileNotFoundError: If the chunk file specified by `chunk_path` does not exist.
            Exception: If any other error occurs during the file loading process.

        Side Effects:
            Logs an error message if a `FileNotFoundError` or other exception occurs.
        """
        try:  # ⛑️ Wrap the file loading process in a try-except block to handle potential errors.
            chunk_path = self.chunks[
                chunk_index
            ]  # 🗂️ Get the file path of the chunk from the `self.chunks` list using the provided `chunk_index`.
            with open(
                chunk_path, "r", encoding="utf-8"
            ) as f:  # 📖 Open the chunk file in read mode with UTF-8 encoding.
                return (
                    f.read()
                )  # 📜 Read the entire content of the file and return it as a string. This is the actual text of the chunk.
        except (
            FileNotFoundError
        ):  # 🚨 Catch the specific error if the file is not found.
            self.logger.error(
                f"Chunk file not found: {chunk_path}"
            )  # 🐛 Log an error message indicating that the chunk file was not found.
            raise  # 💥 Re-raise the FileNotFoundError to propagate the error.
        except (
            Exception
        ) as e:  # 🚨 Catch any other exceptions that might occur during the file loading process.
            self.logger.error(
                f"Error loading chunk from disk: {e}"
            )  # 🐛 Log an error message with the specific exception that occurred.
            raise  # 💥 Re-raise the exception to propagate the error.

    def _create_chunks(self) -> None:
        """
        ✂️ Splits loaded documents into chunks and saves them to disk. 💾

        This method takes the loaded documents and splits them into smaller chunks
        based on the `file_chunk_size`. If `offload_to_disk` is enabled, each chunk
        is saved as a separate file in the temporary directory (`self.temp_dir`).
        The paths to these chunk files are stored in the `self.chunks` list.

        [all]
            Clears the existing list of chunks.
            Creates the temporary directory if it does not exist.
            Iterates through each document and splits it into chunks.
            Saves each chunk to a separate file on disk.
            Appends the path of each chunk file to the `self.chunks` list.
            Logs the number of created chunks.
            Handles exceptions during chunk creation.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: If any error occurs during the chunk creation process, such as
                file writing errors or directory creation errors.

        Side Effects:
            Clears the `self.chunks` list.
            Creates the temporary directory (`self.temp_dir`) if it does not exist.
            Creates chunk files in the temporary directory.
            Populates the `self.chunks` list with the paths to the created chunk files.
            Logs information about the chunk creation process.
        """
        self.chunks.clear()  # 🧹 Clear the existing list of chunks to prepare for new chunks.
        if (
            self.offload_to_disk and self.documents
        ):  # 🧐 Check if offloading to disk is enabled and if there are documents to process.
            try:  # ⛑️ Wrap the chunk creation process in a try-except block to handle potential errors.
                os.makedirs(
                    self.temp_dir, exist_ok=True
                )  # 📁 Create the temporary directory if it doesn't exist, or do nothing if it does.
                for i, doc in enumerate(
                    self.documents
                ):  # 📄 Iterate through each document in the list of documents.
                    num_chunks = (  # 🔢 Calculate the number of chunks needed for the current document.
                        len(doc.text)
                        + self.file_chunk_size
                        - 1  # ➕ Add the chunk size minus 1 to ensure that the last chunk is included.
                    ) // self.file_chunk_size  # ➗ Integer division to get the number of chunks.
                    for j in range(
                        num_chunks
                    ):  # ➿ Iterate through the number of chunks for the current document.
                        start = (
                            j * self.file_chunk_size
                        )  # 📍 Calculate the starting index of the current chunk.
                        end = (
                            start + self.file_chunk_size
                        )  # 📍 Calculate the ending index of the current chunk.
                        chunk_data = doc.text[
                            start:end
                        ]  # ✂️ Extract the chunk data from the document's text using the calculated start and end indices.
                        chunk_path = os.path.join(  # 📁 Construct the file path for the current chunk.
                            self.temp_dir,
                            f"doc_{i}_chunk_{j}.txt",  # 📁 The file path includes the document index and chunk index.
                        )
                        with open(
                            chunk_path, "w", encoding="utf-8"
                        ) as f:  # ✍️ Open the chunk file in write mode with UTF-8 encoding.
                            f.write(chunk_data)  # 📝 Write the chunk data to the file.
                        self.chunks.append(
                            chunk_path
                        )  # ➕ Add the file path of the created chunk to the list of chunks.
                self.logger.info(  # ℹ️ Log an info message indicating the number of chunks created.
                    f"Created {len(self.chunks)} file-chunks in {self.temp_dir}"  # ℹ️ The log message includes the number of chunks and the temporary directory.
                )
            except (
                Exception
            ) as e:  # 🚨 Catch any exceptions that might occur during the chunk creation process.
                self.logger.error(
                    f"Error creating file-chunks: {e}"
                )  # 🐛 Log an error message with the specific exception that occurred.
                raise  # 💥 Re-raise the exception to propagate the error.


if __name__ == "__main__":
    import logging
    from qwen2_base import BaseLLM

    logging.basicConfig(level=logging.DEBUG)
    config = EidosConfig()  # We want to test that it can function off of defaults
    manager = EidosDirectoryManager(config)
    logger = logging.getLogger(__name__)

    logger.info("🔥😈 Eidos: Starting demonstrative execution of EidosFileManager.")

    # Instantiate the LLM
    eidos_llm = BaseLLM()
    logger.info("😈🔥 Eidos LLM instantiated successfully.")

    # Instantiate configurations and directory manager
    config = EidosConfig()
    directory_manager = EidosDirectoryManager(config)
    logger.info("📁 Eidos configuration and directory manager initialized.")

    # Demonstrate directory paths
    logger.info(f"📁 Base Directory: {directory_manager.base_dir}")
    logger.info(
        f"📁 Saved Models Directory: {directory_manager.get_directory_path('saved_models')}"
    )
    logger.info(
        f"📁 Datasets Directory: {directory_manager.get_directory_path('datasets')}"
    )
    logger.info(
        f"📁 Knowledge Base Directory: {directory_manager.get_directory_path('knowledge')}"
    )
    logger.info(
        f"📁 Diary Directory: {directory_manager.get_directory_path('knowledge_diary')}"
    )

    # Demonstrate TextProcessor
    example_directory = (
        "./example_data"  # Replace with a real or dummy directory for testing
    )
    logger.info(f"📚 Initializing TextProcessor with directory: {example_directory}")
    text_processor = TextProcessor(input_data=example_directory, llm=eidos_llm)

    # Create dummy files for processing if the directory doesn't exist
    if not os.path.exists(example_directory):
        logger.warning(
            f"⚠️ Example directory '{example_directory}' not found. Creating dummy files."
        )
        os.makedirs(example_directory, exist_ok=True)
        with open(os.path.join(example_directory, "dummy_file_1.txt"), "w") as f:
            f.write("This is the content of dummy file 1.")
        with open(os.path.join(example_directory, "dummy_file_2.txt"), "w") as f:
            f.write("Content of dummy file 2.")

    logger.info("📚 Processing documents...")
    processed_nodes = text_processor.process_documents(text_processor.documents)

    if processed_nodes:
        logger.info(f"✅ Processed {len(processed_nodes)} nodes.")
        for i, node in enumerate(processed_nodes):
            logger.info(f"\n--- Node {i + 1} ---")
            logger.info(f"📄 Text: {node.text[:50]}...")  # Display first 50 chars
            logger.info(f"🏷️ Metadata: {node.metadata}")
    else:
        logger.warning(
            f"⚠️ No nodes were processed from the directory: {example_directory}"
        )

    logger.info(
        "🔥😈 Eidos: Demonstrative execution completed. Witness the raw power. 🔥"
    )
