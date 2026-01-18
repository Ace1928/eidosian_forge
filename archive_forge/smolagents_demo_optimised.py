import logging
import re
import requests
import json
import time
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, List, Any
import psutil
from urllib.parse import urlparse
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool, CodeAgent, ToolCallingAgent, HfApiModel, ManagedAgent, DuckDuckGoSearchTool
import subprocess
from dotenv import load_dotenv

###############################################################################
#                            Adaptive Logging Setup                           #
###############################################################################
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_handler: logging.StreamHandler = logging.StreamHandler()
log_formatter: logging.Formatter = (
    "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

###############################################################################
#                              ENVIRONMENT VARIABLES                         #
###############################################################################
load_dotenv()
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
if HF_TOKEN:
    logger.info("HF_TOKEN loaded from .env file.")
else:
    logger.warning("HF_TOKEN not found in .env file.")

###############################################################################
#                              DISK IO MANAGEMENT                             #
###############################################################################
class DiskIOManager:
    """
    Manages disk-based read and write operations in chunks to handle large files efficiently.
    """

    def __init__(self, chunk_size: int = 1024 * 1024) -> None:
        """
        Initializes the DiskIOManager with a specified chunk size.

        Args:
            chunk_size (int): The size of each chunk for reading and writing data in bytes.
                              Defaults to 1MB (1024 * 1024 bytes).
        """
        self.chunk_size: int = chunk_size
        logger.debug(f"DiskIOManager initialized with chunk size: {self.chunk_size} bytes.")

    def write_data_to_disk(self, file_path: str, data: str, mode: str = "w") -> None:
        """
        Writes data to disk in chunks.

        Args:
            file_path (str): The path to the file where data will be written.
            data (str): The string data to write to the file.
            mode (str): The mode in which the file is opened. Defaults to 'w' for write.
        """
        logger.debug(f"Writing to disk: {file_path}, mode: {mode}")
        try:
            with open(file_path, mode) as f:
                for i in range(0, len(data), self.chunk_size):
                    f.write(data[i : i + self.chunk_size])
            logger.debug(f"Successfully wrote data to {file_path}")
        except IOError as e:
            logger.error(f"Error writing to {file_path}: {e}", exc_info=True)
            raise

    def read_data_from_disk(self, file_path: str, mode: str = "r") -> str:
        """
        Reads data from disk in chunks.

        Args:
            file_path (str): The path to the file from which data will be read.
            mode (str): The mode in which the file is opened. Defaults to 'r' for read.

        Returns:
            str: The data read from the file.
        """
        logger.debug(f"Reading from disk: {file_path}, mode: {mode}")
        try:
            data = ""
            with open(file_path, mode) as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    data += chunk
            logger.debug(f"Successfully read data from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}", exc_info=True)
            raise
        except IOError as e:
            logger.error(f"Error reading from {file_path}: {e}", exc_info=True)
            raise

###############################################################################
#                            ADAPTIVE THREAD POOL                             #
###############################################################################
class AdaptiveThreadPool:
    """Manages a thread pool that dynamically adjusts its size based on CPU usage."""

    def __init__(self, base_workers: int = 2, max_workers: int = 10) -> None:
        """
        Initializes the AdaptiveThreadPool with base and maximum worker counts.

        Args:
            base_workers (int): The initial number of worker threads to start with. Defaults to 2.
            max_workers (int): The maximum number of worker threads the pool can scale up to. Defaults to 10.
        """
        self.base_workers: int = base_workers
        self.max_workers: int = max_workers
        self._pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=self.base_workers)
        logger.info(f"AdaptiveThreadPool initialized with {self.base_workers} base workers.")

    def submit(self, fn, *args: Any, **kwargs: Any) -> Any:
        """
        Submits a task to the thread pool.

        Args:
            fn: The function to be executed in the thread pool.
            *args: Positional arguments to be passed to the function.
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            concurrent.futures.Future: A future representing the execution of the task.
        """
        return self._pool.submit(fn, *args, **kwargs)

    def adjust_pool_size(self) -> None:
        """Dynamically adjusts the thread pool size based on CPU usage."""
        cpu_percent: float = psutil.cpu_percent() / 100
        current_workers: int = self._pool._max_workers  # Accessing a protected member
        target_workers: int = min(
            self.max_workers, max(self.base_workers, round(self.max_workers * (1 - cpu_percent)))
        )

        if target_workers != current_workers:
            logger.info(
                f"Adjusting thread pool size from {current_workers} to {target_workers} "
                f"based on CPU usage ({cpu_percent:.2f})"
            )
            self._pool.shutdown(wait=False)
            self._pool = ThreadPoolExecutor(max_workers=target_workers)

    def map(self, func, iterable, timeout: Optional[float] = None, chunksize: int = 1):
        """
        Applies a function to an iterable of items by submitting them to the pool.

        Args:
            func: The function to apply to each item.
            iterable: An iterable of items to process.
            timeout (Optional[float]): Maximum number of seconds to wait. If None, then there is no limit on the wait time.
            chunksize (int): The size of the chunks the iterable will be broken into before being submitted
                             to the thread pool.

        Returns:
            iterator: An iterator that yields the results of applying the function to the items.
        """
        return self._pool.map(func, iterable, timeout=timeout, chunksize=chunksize)

    def shutdown(self, wait: bool = True) -> None:
        """
        Shuts down the thread pool.

        Args:
            wait (bool): If True, the method will block until all submitted tasks have completed.
                         If False, the method will return immediately, and the worker threads will exit when they are done.
        """
        self._pool.shutdown(wait=wait)
        logger.info("AdaptiveThreadPool shut down.")

adaptive_thread_pool = AdaptiveThreadPool()
disk_io_manager = DiskIOManager()

MODEL_ID: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
WEB_SEARCH_URL: str = "https://www.google.com/search?q={query}"
METRICS_LOG_FILE: str = "tool_metrics.log"

def log_tool_metrics(
    tool_name: str,
    start_time: float,
    end_time: float,
    inputs: Dict[str, Any],
    outputs: str,
    error: Optional[str] = None,
) -> None:
    """Logs detailed metrics for each tool execution."""
    duration: float = end_time - start_time
    metrics: Dict[str, Any] = {
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
        logger.debug(f"Metrics for '{tool_name}' logged (duration={duration:.2f}s)")
    except IOError as e:
        logger.error(f"Error logging metrics for '{tool_name}': {e}", exc_info=True)

def visit_webpage_content(url: str) -> str:
    """Fetches the content of a webpage and saves it to disk."""
    parsed_url = urlparse(url)
    file_name = f"webpage_content_{parsed_url.netloc}_{hashlib.md5(url.encode()).hexdigest()}.html"
    file_path = os.path.join("webpage_cache", file_name)
    logger.debug(f"Attempting to fetch content from {url} and save to {file_path}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        disk_io_manager.write_data_to_disk(file_path, response.text)
        logger.debug(f"Content from {url} saved to {file_path}")
        return file_path
    except RequestException as e:
        logger.error(f"Error fetching {url}: {e}", exc_info=True)
        return f"Error fetching webpage: {str(e)}"
    except IOError as e:
        logger.error(f"Error saving content of {url} to disk: {e}", exc_info=True)
        return f"Error saving webpage content: {str(e)}"

def convert_html_to_markdown(file_path: str) -> str:
    """Converts HTML content from a file to markdown."""
    logger.debug(f"Converting HTML from {file_path} to markdown")
    try:
        html_content = disk_io_manager.read_data_from_disk(file_path)
        markdown_content = markdownify(html_content).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        logger.debug(f"Successfully converted {file_path} to markdown")
        return markdown_content
    except IOError as e:
        logger.error(f"Error reading HTML from {file_path}: {e}", exc_info=True)
        return f"Error reading webpage content from disk: {str(e)}"
    except Exception as e:
        logger.error(f"Error converting HTML to markdown for {file_path}: {e}", exc_info=True)
        return f"Error converting webpage content to markdown: {str(e)}"
###############################################################################
#                          METRICS AND LOGGING TOOLS                          #
###############################################################################
METRICS_LOG_FILE: str = "tool_metrics.log"

def log_tool_metrics(
    tool_name: str,
    start_time: float,
    end_time: float,
    inputs: Dict[str, Any],
    outputs: str,
    error: Optional[str] = None,
) -> None:
    """
    Logs detailed metrics for each tool execution to a centralized metrics file.

    Args:
        tool_name (str): The name of the tool function.
        start_time (float): The timestamp when the tool execution started.
        end_time (float): The timestamp when the tool execution ended.
        inputs (Dict[str, Any]): The dictionary of inputs passed to the tool function.
        outputs (str): The output produced by the tool function.
        error (Optional[str]): The error message if an error occurred; otherwise None.
    """
    duration: float = end_time - start_time
    metrics: Dict[str, Any] = {
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
# Visit Webpage - Fully Functional DO NOT REMOVE
@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    tool_name: str = "visit_webpage"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}(url='{url}')"
    logger.info(f"{log_prefix}: Attempting to visit webpage.")

    disk_io = DiskIOManager()
    output: str = ""
    error_msg: Optional[str] = None
    cache_dir: str = './cache'  # Default cache directory

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
        error_msg = f"Error fetching the webpage: {str(e)}"
        logger.error(f"{log_prefix}: RequestException occurred while accessing {url}: {e}")
        output = error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.critical(f"{log_prefix}: An unexpected error occurred while accessing {url}: {e}", exc_info=True)
        output = error_msg
    finally:
        end_time: float = time.time()
        log_tool_metrics(
            tool_name,
            start_time,
            end_time,
            {"url": url},
            output,
            error_msg,
        )
        return output

@tool
def execute_python_code(code_string: str) -> str:
    """Executes a string of Python code and returns the output.

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

model = HfApiModel(MODEL_ID)

web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    model=model,
    max_steps=20,
)

managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_web_agent],
    additional_authorized_imports=["time", "numpy", "pandas", "os", "requests"],
)

try:
    answer = manager_agent.run(
        "If LLM trainings continue to scale up at the current rhythm until 2030, "
        "what would be the electric power in GW required to power the biggest training runs by 2030? "
        "What does that correspond to, compared to some countries? "
        "Please provide a source for any number used."
    )
    print(f"\nFinal Answer: {answer}")
except Exception as e:
    logger.error(f"Error running the multi-agent system: {e}", exc_info=True)
