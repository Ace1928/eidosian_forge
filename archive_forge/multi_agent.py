from itertools import tee
import os
import logging
import re
import requests
import json
import abc
import hashlib
from urllib.parse import urlparse
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Iterator, List
import psutil
from collections import deque
import time
import subprocess


###############################################################################
#                            Adaptive Logging Setup                           #
###############################################################################
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_handler: logging.StreamHandler = logging.StreamHandler()
log_formatter: logging.Formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

###############################################################################
#                              DISK IO MANAGEMENT                             #
###############################################################################
class DiskIOManager:
    """
    Manages disk-based read and write operations in chunks to handle large files efficiently.
    
    This class provides methods for writing data to disk, reading data from disk,
    and reading data from disk in chunks, optimizing for memory usage and scalability.
    """

    def __init__(self, chunk_size: int = 1024 * 1024) -> None:
        """
        Initializes the DiskIOManager with a specified chunk size.

        Args:
            chunk_size (int): The size of each chunk for read/write operations in bytes.
                              Defaults to 1MB.
        """
        self.chunk_size: int = chunk_size
        logger.debug(f"DiskIOManager initialized with chunk size: {self.chunk_size} bytes.")

    def write_data_to_disk(self, file_path: str, data: str, mode: str = "w") -> None:
        """
        Writes data to disk in chunks to handle large strings efficiently.

        Args:
            file_path (str): The path to the file where data will be written.
            data (str): The string data to write to the file.
            mode (str): The file opening mode (e.g., 'w' for write, 'a' for append).
                        Defaults to 'w'.

        Raises:
            IOError: If an error occurs during the file writing process.
        """
        logger.debug(
            f"Writing data to disk in chunks: file_path='{file_path}', mode='{mode}', "
            f"chunk_size='{self.chunk_size}'"
        )
        try:
            with open(file_path, mode, encoding="utf-8") as f:
                for i in range(0, len(data), self.chunk_size):
                    chunk: str = data[i : i + self.chunk_size]
                    f.write(chunk)
            logger.info(f"Successfully wrote data to disk: {file_path}")
        except IOError as e:
            logger.error(f"Error writing to file '{file_path}': {e}", exc_info=True)
            raise

    def read_data_from_disk(self, file_path: str, mode: str = "r") -> str:
        """
        Reads data from disk in chunks and returns the entire content as a string.

        Args:
            file_path (str): The path to the file to read from.
            mode (str): The file opening mode (e.g., 'r' for read).
                        Defaults to 'r'.

        Returns:
            str: The entire content of the file.
        
        Raises:
            IOError: If an error occurs during the file reading process.
        """
        logger.debug(
            f"Reading data from disk in chunks: file_path='{file_path}', mode='{mode}', "
            f"chunk_size='{self.chunk_size}'"
        )
        content: str = ""
        try:
            with open(file_path, mode, encoding="utf-8") as f:
                while True:
                    chunk: str = f.read(self.chunk_size)
                    if not chunk:
                        break
                    content += chunk
            logger.info(f"Successfully read data from disk: {file_path}")
            return content
        except IOError as e:
            logger.error(f"Error reading from file '{file_path}': {e}", exc_info=True)
            raise

    def read_data_from_disk_in_chunks(
        self, file_path: str, mode: str = "r"
    ) -> Iterator[str]:
        """
        Reads data from disk in chunks and yields each chunk as an iterator.
        This method is memory-efficient for processing large files,
        as it does not load the entire file into memory at once.

        Args:
            file_path (str): The path to the file to read from.
            mode (str): The file opening mode (e.g., 'r' for read). Defaults to 'r'.

        Yields:
            str: A chunk of data read from the file.
        
        Raises:
            IOError: If an error occurs during the file reading process.
        """
        logger.debug(
            f"Reading data from disk in chunks (iterator): file_path='{file_path}', "
            f"mode='{mode}', chunk_size='{self.chunk_size}'"
        )
        try:
            with open(file_path, mode, encoding="utf-8") as f:
                while True:
                    chunk: str = f.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk
            logger.info(f"Finished reading data from disk in chunks (iterator): {file_path}")
        except IOError as e:
            logger.error(f"Error reading from file '{file_path}': {e}", exc_info=True)
            raise

###############################################################################
#                           RESOURCE MANAGEMENT                               #
###############################################################################
class ResourceManager:
    """
    Manages and monitors system resources (CPU, disk, memory) for adaptive processing.
    
    This class tracks resource usage and provides methods to determine if concurrency
    should be reduced based on historical resource consumption.
    """

    def __init__(self) -> None:
        """
        Initializes the ResourceManager with default thresholds and history size.
        """
        self.cpu_threshold: float = 80.0  # CPU usage threshold (%)
        self.disk_threshold: float = 80.0  # Disk usage threshold (%)
        self.memory_threshold: float = 80.0  # Memory usage threshold (%)
        self.history_size: int = 10  # Number of recent measurements to consider

        self.cpu_history: deque[float] = deque(maxlen=self.history_size)
        self.disk_history: deque[float] = deque(maxlen=self.history_size)
        self.memory_history: deque[float] = deque(maxlen=self.history_size)

        logger.debug("ResourceManager initialized.")

    def get_resource_usage(self) -> Dict[str, float]:
        """
        Returns current CPU, disk, and memory usage as a dictionary.

        Returns:
            Dict[str, float]: A dictionary containing CPU, disk, and memory usage percentages.
        """
        cpu_percent: float = psutil.cpu_percent()
        disk_usage: float = psutil.disk_usage("/").percent
        memory_usage: float = psutil.virtual_memory().percent

        self.cpu_history.append(cpu_percent)
        self.disk_history.append(disk_usage)
        self.memory_history.append(memory_usage)

        logger.debug(
            f"Current resource usage: CPU={cpu_percent}%, "
            f"Disk={disk_usage}%, Memory={memory_usage}%"
        )
        return {
            "cpu_percent": cpu_percent,
            "disk_percent": disk_usage,
            "memory_percent": memory_usage,
        }

    def should_reduce_concurrency(self) -> bool:
        """
        Determines if concurrency should be reduced based on the average resource usage history.

        Returns:
            bool: True if concurrency should be reduced, False otherwise.
        """
        avg_cpu: float = (
            sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0
        )
        avg_disk: float = (
            sum(self.disk_history) / len(self.disk_history) if self.disk_history else 0
        )
        avg_memory: float = (
            sum(self.memory_history) / len(self.memory_history)
            if self.memory_history
            else 0
        )

        should_reduce: bool = (
            avg_cpu > self.cpu_threshold
            or avg_disk > self.disk_threshold
            or avg_memory > self.memory_threshold
        )
        logger.debug(
            f"Average resource usage: CPU={avg_cpu}%, Disk={avg_disk}%, "
            f"Memory={avg_memory}%. Reduce concurrency: {should_reduce}"
        )
        return should_reduce

###############################################################################
#                         ADAPTIVE THREAD POOL EXECUTOR                       #
###############################################################################
class AdaptiveThreadPoolExecutor(ThreadPoolExecutor):
    """
    A thread pool executor that dynamically adjusts the number of worker threads
    based on system resource usage.

    This class helps in optimizing resource utilization by scaling the number of threads
    up or down as needed.
    """

    def __init__(
        self,
        initial_max_workers: int,
        resource_manager: ResourceManager,
        min_workers: int = 1,
        max_workers_limit: Optional[int] = None,
    ) -> None:
        """
        Initializes the AdaptiveThreadPoolExecutor.

        Args:
            initial_max_workers (int): The initial maximum number of worker threads.
            resource_manager (ResourceManager): An instance of ResourceManager to monitor system resources.
            min_workers (int): The minimum number of worker threads. Defaults to 1.
            max_workers_limit (Optional[int]): A hard limit for the maximum number of
                                               worker threads. If None, defaults to
                                               initial_max_workers.
        """
        super().__init__(max_workers=initial_max_workers)
        self.resource_manager: ResourceManager = resource_manager
        self.initial_max_workers: int = initial_max_workers
        self.min_workers: int = min_workers
        self.max_workers_limit: Optional[int] = max_workers_limit
        self._last_resize_time: float = time.time()
        self._resize_interval: int = 5  # seconds

        logger.debug(
            f"AdaptiveThreadPoolExecutor initialized with initial_max_workers={initial_max_workers}, "
            f"min_workers={min_workers}, max_workers_limit={max_workers_limit}"
        )

    def adjust_pool_size(self) -> None:
        """
        Dynamically adjusts the pool size based on resource usage.
        
        Checks the resource manager and either increases or decreases the number of worker
        threads based on current/historical resource consumption.
        """
        if time.time() - self._last_resize_time < self._resize_interval:
            return

        if self.resource_manager.should_reduce_concurrency():
            if self._max_workers > self.min_workers:
                self._max_workers = max(self.min_workers, self._max_workers // 2)
                logger.warning(
                    f"Reducing thread pool size to {self._max_workers} due to high resource usage."
                )
                self._resize_pool()
                self._last_resize_time = time.time()
        else:
            max_limit: int = (
                self.max_workers_limit
                if self.max_workers_limit is not None
                else self.initial_max_workers
            )
            if self._max_workers < max_limit:
                target_workers: int = min(max_limit, self._max_workers * 2)
                if target_workers > self._max_workers:
                    self._max_workers = target_workers
                    logger.info(f"Increasing thread pool size to {self._max_workers}.")
                    self._resize_pool()
                    self._last_resize_time = time.time()

    def _resize_pool(self) -> None:
        """
        Internal method to resize the thread pool by creating new threads or stopping existing ones.
        
        Note: This is a simplified approach. A more robust solution might involve creating
        a new executor and transferring tasks.
        """
        all_threads = list(self._threads)
        for thread in all_threads:
            if thread.is_alive():
                try:
                    thread.join(timeout=0.1)
                except Exception as e:
                    logger.error(f"Error while trying to join thread: {e}", exc_info=True)

        self._threads.clear()
        self._work_queue.queue.clear()
        for _ in range(self._max_workers):
            self._start_thread()
        logger.debug(f"Thread pool resized to {self._max_workers} workers.")

###############################################################################
#                           ENVIRONMENT CONFIGURATION                         #
###############################################################################
class EnvironmentConfig:
    """
    Handles environment configuration and directory setup with disk-based logging.

    This class manages settings like the Hugging Face token and creates necessary
    directories for the application.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        directories: Optional[Dict[str, str]] = None,
        disk_io_manager: Optional[DiskIOManager] = None,
    ) -> None:
        """
        Initializes the EnvironmentConfig.

        Args:
            hf_token (Optional[str]): The Hugging Face API token. Defaults to the
                                      'HF_TOKEN' environment variable or a fallback.
            directories (Optional[Dict[str, str]]): A dictionary of directory names and their paths.
            disk_io_manager (Optional[DiskIOManager]): An instance of DiskIOManager for disk operations.
        """
        self.hf_token: str = hf_token or os.getenv(
            "HF_TOKEN", "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"
        )
        self.directories: Dict[str, str] = directories or {
            "saved_model": "./saved_models",
            "datasets": "./datasets",
            "checkpoints": "./checkpoints",
            "cache": "./cache",
            "logs": "./logs",
            "offload": "./offload",
            "memory": "./memory",
            "code_library": "./code_library",
            "documents": "./documents",
            "images": "./images",
            "audio": "./audio",
            "video": "./video",
        }
        self.disk_io_manager: Optional[DiskIOManager] = disk_io_manager
        self._validate_environment()
        self._create_directories()
        logger.debug("EnvironmentConfig initialized.")

    def _validate_environment(self) -> None:
        """
        Validates the environment configuration, logging warnings for missing settings.
        """
        if not self.hf_token:
            logger.warning("HF_TOKEN environment variable is not set. Using default token.")

    def _create_directories(self) -> None:
        """
        Creates necessary directories if they do not exist, leveraging disk I/O.
        """
        logger.info("Starting directory creation process.")
        for dir_name, path in self.directories.items():
            self._create_directory(dir_name, path)
        logger.info("Directory creation process completed.")

    def _create_directory(self, dir_name: str, path: str) -> None:
        """
        Creates a single directory with detailed logging and error handling.

        Args:
            dir_name (str): The directory name.
            path (str): The path to the directory.
        """
        if not os.path.exists(path):
            logger.debug(f"Directory '{dir_name}' does not exist. Attempting to create: {path}")
            try:
                os.makedirs(path, exist_ok=True)
                logger.info(f"Successfully created directory: {path}")
            except OSError as e:
                logger.error(
                    f"Failed to create directory '{dir_name}' at '{path}': {e}",
                    exc_info=True,
                )
                raise
        else:
            logger.debug(f"Directory '{dir_name}' already exists: {path}")

###############################################################################
#                              MODEL MANAGEMENT                               #
###############################################################################
class ModelManager:
    """
    Manages downloading and saving models from Hugging Face, prioritizing disk-based
    operations and utilizing adaptive concurrency.
    """

    def __init__(
        self,
        disk_io_manager: Optional[DiskIOManager] = None,
        resource_manager: Optional[ResourceManager] = None,
        thread_pool: Optional[AdaptiveThreadPoolExecutor] = None,
    ) -> None:
        """
        Initializes the ModelManager.

        Args:
            disk_io_manager (Optional[DiskIOManager]): An instance of DiskIOManager for disk operations.
            resource_manager (Optional[ResourceManager]): An instance of ResourceManager for monitoring system resources.
            thread_pool (Optional[AdaptiveThreadPoolExecutor]): An AdaptiveThreadPoolExecutor for concurrent downloads.
        """
        self.disk_io_manager: DiskIOManager = disk_io_manager or DiskIOManager()
        self.resource_manager: ResourceManager = resource_manager or ResourceManager()
        self.thread_pool: AdaptiveThreadPoolExecutor = (
            thread_pool
            or AdaptiveThreadPoolExecutor(
                initial_max_workers=2, resource_manager=self.resource_manager
            )
        )
        logger.debug("ModelManager initialized.")

    def download_and_save_model(self, model_name: str, save_path: str) -> None:
        """
        Downloads a model and its tokenizer from Hugging Face and saves them locally.

        This process prioritizes saving configurations to disk first and uses adaptive
        concurrency to manage resource usage during download.

        Args:
            model_name (str): The name of the model to download from Hugging Face.
            save_path (str): The local path where the model and tokenizer will be saved.

        Raises:
            Exception: If any error occurs during the download or save process.
        """
        log_prefix: str = f"download_and_save_model(model_name='{model_name}', save_path='{save_path}')"
        logger.info(f"{log_prefix}: Starting model download and save process.")

        try:
            config_path: str = os.path.join(save_path, "config")
            os.makedirs(config_path, exist_ok=True)
            config = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=config_path, local_files_only=False
            )
            config.save_pretrained(config_path)
            logger.debug(f"{log_prefix}: Model configuration saved to disk at {config_path}")

            tokenizer_path: str = os.path.join(save_path, "tokenizer")
            os.makedirs(tokenizer_path, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=tokenizer_path, local_files_only=False
            )
            tokenizer.save_pretrained(tokenizer_path)
            logger.debug(f"{log_prefix}: Tokenizer configuration saved to disk at {tokenizer_path}")

            # Download and save model weights
            model = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=save_path, local_files_only=False
            )
            model.save_pretrained(save_path)
            logger.info(
                f"{log_prefix}: Model and tokenizer successfully downloaded and saved to disk at {save_path}"
            )
        except Exception as e:
            logger.critical(
                f"{log_prefix}: Critical failure during model download and save: {e}",
                exc_info=True,
            )
            raise

###############################################################################
#                              GLOBAL COMPONENTS                               #
###############################################################################
resource_manager: ResourceManager = ResourceManager()
disk_io_manager: DiskIOManager = DiskIOManager()
env_config: EnvironmentConfig = EnvironmentConfig(disk_io_manager=disk_io_manager)
adaptive_thread_pool: AdaptiveThreadPoolExecutor = AdaptiveThreadPoolExecutor(
    initial_max_workers=4, resource_manager=resource_manager
)
model_manager: ModelManager = ModelManager(
    disk_io_manager=disk_io_manager,
    resource_manager=resource_manager,
    thread_pool=adaptive_thread_pool,
)

MODEL_ID: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
WEB_SEARCH_URL: str = "https://www.google.com/search?q={query}"

###############################################################################
#                          METRICS AND LOGGING TOOLS                          #
###############################################################################
METRICS_LOG_FILE: str = "tool_metrics.log"

def log_tool_metrics(
    tool_name: str,
    start_time: float,
    end_time: float,
    inputs: Dict,
    outputs: str,
    error: Optional[str] = None,
) -> None:
    """
    Logs detailed metrics for each tool execution to a centralized metrics file.

    Args:
        tool_name (str): The name of the tool function.
        start_time (float): The timestamp when the tool execution started.
        end_time (float): The timestamp when the tool execution ended.
        inputs (Dict): The dictionary of inputs passed to the tool function.
        outputs (str): The output produced by the tool function.
        error (Optional[str]): The error message if an error occurred; otherwise None.
    """
    duration: float = end_time - start_time
    metrics: Dict = {
        "tool_name": tool_name,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "inputs": inputs,
        "outputs": outputs,
        "error": error,
    }
    try:
        with open(METRICS_LOG_FILE, "a") as f:
            json.dump(metrics, f)
            f.write("\n")
        logger.debug(f"Metrics for '{tool_name}' logged to {METRICS_LOG_FILE} (duration={duration:.2f}s)")
    except IOError as e:
        logger.error(f"Error logging metrics for '{tool_name}': {e}", exc_info=True)

def get_tool_feedback(tool_name: str) -> str:
    """
    Retrieves and processes feedback for a specific tool from the metrics log file.

    Args:
        tool_name (str): The name of the tool for which to retrieve feedback.

    Returns:
        str: A text summary of the feedback, including success rates and errors.
    """
    feedback_content: str = ""
    try:
        with open(METRICS_LOG_FILE, "r") as f:
            all_metrics = [json.loads(line) for line in f if line.strip()]

        tool_metrics = [m for m in all_metrics if m["tool_name"] == tool_name]
        if not tool_metrics:
            return f"No metrics found for tool '{tool_name}'."

        total_executions: int = len(tool_metrics)
        successful_executions: int = sum(1 for m in tool_metrics if not m.get("error"))
        error_rate: float = (
            (1 - successful_executions / total_executions) * 100
            if total_executions > 0
            else 0
        )

        feedback_content += f"Feedback for tool '{tool_name}':\n"
        feedback_content += f"Total executions: {total_executions}\n"
        feedback_content += f"Successful executions: {successful_executions}\n"
        feedback_content += f"Error rate: {error_rate:.2f}%\n"

        if error_rate > 0:
            error_messages = [m["error"] for m in tool_metrics if m.get("error")]
            feedback_content += "Common errors:\n"
            for error in set(error_messages):
                feedback_content += f"- {error}\n"

    except FileNotFoundError:
        return f"Metrics log file '{METRICS_LOG_FILE}' not found."
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from metrics log: {e}", exc_info=True)
        return f"Error reading metrics for tool '{tool_name}'."
    except Exception as e:
        logger.error(f"Unexpected error getting feedback for tool '{tool_name}': {e}", exc_info=True)
        return f"Error processing feedback for tool '{tool_name}'."

    return feedback_content

###############################################################################
#                               TOOL FUNCTIONS                                #
###############################################################################
@tool
def visit_webpage(
    url: str,
    cache_dir: Optional[str] = './cache',
) -> str:
    """
    Visits a webpage, saves the raw content to disk, converts it to markdown,
    and returns the markdown-formatted string.

    Args:
        url: The URL of the webpage to visit.
        cache_dir: The directory where the raw webpage content will be saved. Defaults to './cache'.
                   (nullable: True)

    Returns:
        The markdown-formatted content of the webpage, or an error message if fetching fails.
    """
    tool_name: str = "visit_webpage"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}(url='{url}')"
    logger.info(f"{log_prefix}: Attempting to visit webpage.")

    disk_io = DiskIOManager()
    output: str = ""
    error_msg: Optional[str] = None
    try:
        response: requests.Response = requests.get(url, timeout=10)
        response.raise_for_status()

        raw_content_path: str = os.path.join(
            cache_dir,
            f"{urlparse(url).netloc}_{hashlib.md5(url.encode()).hexdigest()[:8]}.html",
        )
        os.makedirs(cache_dir, exist_ok=True)
        disk_io.write_data_to_disk(raw_content_path, response.text)
        logger.debug(f"{log_prefix}: Raw content saved to disk: {raw_content_path}")

        markdown_chunks: List[str] = []
        for chunk in disk_io.read_data_from_disk_in_chunks(raw_content_path):
            markdown_chunk: str = markdownify(chunk).strip()
            markdown_chunks.append(markdown_chunk)

        markdown_content: str = "\n\n".join(markdown_chunks)
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        output = markdown_content

        logger.info(f"{log_prefix}: Successfully fetched and converted webpage content.")
    except RequestException as e:
        error_msg = f"Error fetching the webpage: {e}"
        logger.error(f"{log_prefix}: RequestException occurred while accessing {url}: {e}")
        output = error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        logger.critical(f"{log_prefix}: An unexpected error occurred while accessing {url}: {e}", exc_info=True)
        output = error_msg
    finally:
        end_time: float = time.time()
        log_tool_metrics(
            tool_name,
            start_time,
            end_time,
            {"url": url, "cache_dir": cache_dir},
            output,
            error_msg,
        )
        return output

@tool
def execute_python_code(code_string: str) -> str:
    """
    Executes a string of Python code and returns the output.

    Args:
        code_string: The Python code to execute.

    Returns:
        The output of the executed code, or an error message if an error occurred.
    """
    tool_name: str = "execute_python_code"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}(code_string='{code_string[:50]}...')"
    logger.info(f"{log_prefix}: Attempting to execute Python code.")
    output: str = ""
    error_msg: Optional[str] = None

    try:
        process = subprocess.Popen(
            ["python", "-c", code_string],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(timeout=15)
        if stderr:
            error_msg = f"Error executing code:\n{stderr}"
            logger.error(f"{log_prefix}: Error executing code: {stderr}")
            output = error_msg
        else:
            output = stdout
            logger.info(f"{log_prefix}: Successfully executed code.")
    except subprocess.TimeoutExpired:
        error_msg = "Error: Code execution timed out."
        logger.error(f"{log_prefix}: Code execution timed out.")
        output = error_msg
    except Exception as e:
        error_msg = f"Error executing code: {e}"
        logger.error(f"{log_prefix}: Unexpected error executing code: {e}", exc_info=True)
        output = error_msg
    finally:
        end_time: float = time.time()
        log_tool_metrics(
            tool_name,
            start_time,
            end_time,
            {"code_string": code_string},
            output,
            error_msg,
        )
        return output

@tool
def read_file(
    file_path: str,
) -> str:
    """
    Reads the content of a file from disk.

    Args:
        file_path: The path to the file to read.

    Returns:
        The content of the file, or an error message if an error occurred.
    """
    tool_name: str = "read_file"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}(file_path='{file_path}')"
    logger.info(f"{log_prefix}: Attempting to read file.")
    disk_io = DiskIOManager()
    output: str = ""
    error_msg: Optional[str] = None

    try:
        content = disk_io.read_data_from_disk(file_path, mode="r")
        output = content
        logger.info(f"{log_prefix}: Successfully read file.")
    except IOError as e:
        error_msg = f"Error reading file: {e}"
        logger.error(f"{log_prefix}: Error reading file: {e}", exc_info=True)
        output = error_msg
    finally:
        end_time: float = time.time()
        log_tool_metrics(
            tool_name, start_time, end_time, {"file_path": file_path}, output, error_msg
        )
        return output

@tool
def write_file(
    file_path: str,
    content: str,
    mode: Optional[str] = 'w',
) -> None:
    """
    Writes content to a file on disk.

    Args:
        file_path: The path to the file to write.
        content: The text content to write to the file.
        mode: The file mode (e.g., 'w' for write, 'a' for append). Defaults to 'w'. (nullable: True)

    Returns:
        None
    """
    tool_name: str = "write_file"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}(file_path='{file_path}', mode='{mode}')"
    logger.info(f"{log_prefix}: Attempting to write to file.")
    disk_io = DiskIOManager()

    output: str = ""
    error_msg: Optional[str] = None
    try:
        disk_io.write_data_to_disk(file_path, content, mode=mode)
        logger.info(f"{log_prefix}: Successfully wrote to file.")
    except IOError as e:
        error_msg = f"Error writing to file: {e}"
        logger.error(f"{log_prefix}: Error writing to file: {e}", exc_info=True)
        raise
    finally:
        end_time: float = time.time()
        log_tool_metrics(
            tool_name,
            start_time,
            end_time,
            {"file_path": file_path, "content": content, "mode": mode},
            output,
            error_msg,
        )

@tool
def search_internet(query: str) -> str:
    """
    Searches the internet for the given query using a simplified approach.

    Args:
        query: The search query.

    Returns:
        A summary of the search results or an error message if an error occurred.
    """
    tool_name: str = "search_internet"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}(query='{query}')"
    logger.info(f"{log_prefix}: Attempting to search the internet.")

    output: str = ""
    error_msg: Optional[str] = None
    try:
        from duckduckgo_search import ddg
        results = ddg(query, max_results=5)
        if results:
            summary = "\n".join(
                [
                    f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}"
                    for r in results
                ]
            )
            output = summary
            logger.info(f"{log_prefix}: Successfully retrieved search results.")
        else:
            output = "No relevant search results found."
            logger.info(f"{log_prefix}: No relevant search results found.")
    except ImportError:
        error_msg = "Error: The 'duckduckgo-search' library is required for this tool."
        logger.error(f"{log_prefix}: duckduckgo-search library is not installed.")
        output = error_msg
    except Exception as e:
        error_msg = f"Error during internet search: {e}"
        logger.error(f"{log_prefix}: Error during internet search: {e}", exc_info=True)
        output = error_msg
    finally:
        end_time: float = time.time()
        log_tool_metrics(
            tool_name, start_time, end_time, {"query": query}, output, error_msg
        )
        return output

@tool
def list_files_in_directory(
    path: str = '.',
) -> str:
    """
    Lists all files and directories in the specified path.

    Args:
        path: The directory path to list. Defaults to '.' (current directory). (nullable: True)

    Returns:
        A list of files and directories, one per line, or an error message if an error occurred.
    """
    tool_name: str = "list_files_in_directory"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}(path='{path}')"
    logger.info(f"{log_prefix}: Attempting to list files in directory.")

    output: str = ""
    error_msg: Optional[str] = None
    try:
        items = os.listdir(path)
        output = "\n".join(items)
        logger.info(f"{log_prefix}: Successfully listed files in directory: {path}")
    except FileNotFoundError:
        error_msg = f"Error: Directory not found: {path}"
        logger.error(f"{log_prefix}: Directory not found: {path}")
        output = error_msg
    except Exception as e:
        error_msg = f"Error listing directory: {e}"
        logger.error(f"{log_prefix}: Error listing directory: {e}", exc_info=True)
        output = error_msg
    finally:
        end_time: float = time.time()
        log_tool_metrics(tool_name, start_time, end_time, {"path": path}, output, error_msg)
        return output

@tool
def get_file_metadata(file_path: str) -> str:
    """
    Retrieves metadata for a given file.

    Args:
        file_path: The path to the file whose metadata should be retrieved.

    Returns:
        A JSON-formatted string containing file metadata, or an error message if an error occurred.
    """
    tool_name: str = "get_file_metadata"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}(file_path='{file_path}')"
    logger.info(f"{log_prefix}: Attempting to get file metadata.")

    output: str = ""
    error_msg: Optional[str] = None
    try:
        metadata = {
            "size": os.path.getsize(file_path),
            "modified_time": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(file_path))
            ),
        }
        output = json.dumps(metadata, indent=4)
        logger.info(f"{log_prefix}: Successfully retrieved metadata for: {file_path}")
    except FileNotFoundError:
        error_msg = f"Error: File not found: {file_path}"
        logger.error(f"{log_prefix}: File not found: {file_path}")
        output = error_msg
    except Exception as e:
        error_msg = f"Error getting file metadata: {e}"
        logger.error(f"{log_prefix}: Error getting file metadata: {e}", exc_info=True)
        output = error_msg
    finally:
        end_time: float = time.time()
        log_tool_metrics(
            tool_name, start_time, end_time, {"file_path": file_path}, output, error_msg
        )
        return output

@tool
def download_file(
    url: str,
    save_path: str,
) -> str:
    """
    Downloads a file from the specified URL and saves it to the given path.

    Args:
        url: The URL of the file to download.
        save_path: The local path where the file will be saved.

    Returns:
        A message indicating success or an error message if an error occurred.
    """
    tool_name: str = "download_file"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}(url='{url}', save_path='{save_path}')"
    logger.info(f"{log_prefix}: Attempting to download file.")

    disk_io = DiskIOManager()
    output: str = ""
    error_msg: Optional[str] = None
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=disk_io.chunk_size):
                file.write(chunk)
        output = f"File successfully downloaded to: {save_path}"
        logger.info(f"{log_prefix}: File successfully downloaded to: {save_path}")
    except RequestException as e:
        error_msg = f"Error downloading file: {e}"
        logger.error(f"{log_prefix}: Error downloading file: {e}")
        output = error_msg
    except IOError as e:
        error_msg = f"Error saving downloaded file: {e}"
        logger.error(f"{log_prefix}: Error saving downloaded file: {e}", exc_info=True)
        output = error_msg
    except Exception as e:
        error_msg = f"Unexpected error during file download: {e}"
        logger.error(f"{log_prefix}: Unexpected error during file download: {e}", exc_info=True)
        output = error_msg
    finally:
        end_time: float = time.time()
        log_tool_metrics(
            tool_name,
            start_time,
            end_time,
            {"url": url, "save_path": save_path},
            output,
            error_msg,
        )
        return output

@tool
def make_directory(path: str) -> str:
    """
    Creates a new directory at the specified path.

    Args:
        path: The path where the new directory should be created.

    Returns:
        A message indicating success or an error message if creation failed.
    """
    tool_name: str = "make_directory"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}(path='{path}')"
    logger.info(f"{log_prefix}: Attempting to create directory.")

    output: str = ""
    error_msg: Optional[str] = None
    try:
        os.makedirs(path, exist_ok=True)
        output = f"Directory successfully created at: {path}"
        logger.info(f"{log_prefix}: Directory successfully created at: {path}")
    except OSError as e:
        error_msg = f"Error creating directory: {e}"
        logger.error(f"{log_prefix}: Error creating directory: {e}", exc_info=True)
        output = error_msg
    finally:
        end_time: float = time.time()
        log_tool_metrics(
            tool_name, start_time, end_time, {"path": path}, output, error_msg
        )
        return output

@tool
def delete_file(file_path: str) -> str:
    """
    Deletes the file at the specified path.

    Args:
        file_path: The path to the file to delete.

    Returns:
        A message indicating success or an error message if deletion failed.
    """
    tool_name: str = "delete_file"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}(file_path='{file_path}')"
    logger.info(f"{log_prefix}: Attempting to delete file.")

    output: str = ""
    error_msg: Optional[str] = None
    try:
        os.remove(file_path)
        output = f"File successfully deleted: {file_path}"
        logger.info(f"{log_prefix}: File successfully deleted: {file_path}")
    except FileNotFoundError:
        error_msg = f"Error: File not found: {file_path}"
        logger.error(f"{log_prefix}: File not found: {file_path}")
        output = error_msg
    except OSError as e:
        error_msg = f"Error deleting file: {e}"
        logger.error(f"{log_prefix}: Error deleting file: {e}", exc_info=True)
        output = error_msg
    finally:
        end_time: float = time.time()
        log_tool_metrics(
            tool_name, start_time, end_time, {"file_path": file_path}, output, error_msg
        )
        return output

###############################################################################
#                            AGENT OPERATION LOOP                             #
###############################################################################
def agent_loop() -> None:
    """
    Demonstrates an AI agent's autonomous operation and response to user input.

    This function repeatedly performs a predefined action (searching the internet with a sample query),
    retrieves feedback on the tool, and then prompts for user input to either continue autonomously
    or process user input.
    """
    print("Agent is starting...")
    while True:
        action = "search_internet"
        query = "latest AI trends"
        print(f"Agent is performing action: {action} with query: {query}")
        results = search_internet(query)
        print(f"Search results:\n{results}")

        feedback = get_tool_feedback("search_internet")
        print(f"Feedback on search_internet tool:\n{feedback}")

        user_input = input("User input (or type 'continue'): ")
        if user_input.lower() != "continue":
            print(f"Agent responding to user input: {user_input}")
        else:
            print("Agent continuing autonomous operation.")

        time.sleep(10)  # Simulate time passing

###############################################################################
#                               MAIN EXECUTION                                #
###############################################################################
if __name__ == "__main__":
    # Example usage of some tools:
    print(visit_webpage(WEB_SEARCH_URL.format(query="artificial intelligence")))
    print(execute_python_code("print('Hello from executed code!')"))
    print(read_file("smolagent_demo.ipynb"))

    write_file("test_output.txt", "This is a test.")
    print(list_files_in_directory())
    print(search_internet("large language models"))
    print(get_file_metadata("smolagent_demo.ipynb"))
    print(download_file("https://www.example.com", "example.html"))
    print(make_directory("new_directory"))
    print(delete_file("test_output.txt"))

    # Start the agent loop (conceptual demonstration)
    # agent_loop()
