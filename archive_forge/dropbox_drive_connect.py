import os
import dropbox
from dropbox.exceptions import AuthError, ApiError, RateLimitError
from dropbox.oauth import DropboxOAuth2FlowNoRedirect
import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from collections import deque, defaultdict
import colorama
from colorama import Fore, Style
from itertools import groupby
from datetime import datetime
import threading
from dotenv import load_dotenv


load_dotenv()

colorama.init()

# --- Enhanced Configuration with Descriptions and Validation ---
class Config:
    def __init__(self):
        self.TOKEN_FILE = 'dropbox_token.json'
        self.APP_KEY = os.environ.get('DROPBOX_APP_KEY')
        self.APP_SECRET = os.environ.get('DROPBOX_APP_SECRET')
        self.ACCESS_TOKEN = os.environ.get('DROPBOX_ACCESS_TOKEN')
        self.MAX_DEPTH = self._get_env_var('DROPBOX_MAX_DEPTH', default=10, type=int)
        self.LOG_LEVEL = self._get_env_var('LOG_LEVEL', default='INFO').upper()
        self.MAX_WORKERS = self._get_env_var('MAX_WORKERS', default=os.cpu_count(), type=int)
        self.LIST_FOLDER_CHUNK_SIZE = self._get_env_var('LIST_FOLDER_CHUNK_SIZE', default=1000, type=int)
        self.OUTPUT_FILE = self._get_env_var('DROPBOX_OUTPUT_FILE', default='dropbox_folder_structure.json')
        self.METRICS_FILE = self._get_env_var('DROPBOX_METRICS_FILE', default='dropbox_metrics.json')
        self.OUTPUT_FORMAT = self._get_env_var('OUTPUT_FORMAT', default='json').lower()
        self.LOG_FILE = self._get_env_var('LOG_FILE', default='dropbox_app.log')
        self.ERROR_LOG_FILE = self._get_env_var('ERROR_LOG_FILE', default='dropbox_app_error.log')
        self.METRICS_LOG_LEVEL = self._get_env_var('METRICS_LOG_LEVEL', default='INFO').upper()
        self.FILE_INFO_CHUNK_SIZE = self._get_env_var('FILE_INFO_CHUNK_SIZE', default=100, type=int)
        self.REPORT_INTERVAL = self._get_env_var('REPORT_INTERVAL', default=10, type=int)  # Report every N seconds
        self.API_RETRY_DELAY = self._get_env_var('API_RETRY_DELAY', default=5, type=int) # Seconds to wait before retrying API calls
        self.API_MAX_RETRIES = self._get_env_var('API_MAX_RETRIES', default=3, type=int) # Maximum number of retries for API calls

    def _get_env_var(self, key, default=None, required=False, type=str):
        value = os.environ.get(key)
        if value is None:
            if required:
                raise ValueError(f"Environment variable '{key}' is required.")
            return default
        try:
            return type(value)
        except ValueError:
            raise ValueError(f"Invalid type for environment variable '{key}'. Expected {type.__name__}.")

config = Config()

# --- Advanced Logging Setup with Rotating File Handlers and Detailed Formatting ---
class LoggerManager:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)
        self.metrics_logger = self._setup_logger('metrics', config.METRICS_FILE, config.METRICS_LOG_LEVEL)

    def _setup_logger(self, name: str, log_file: str, level: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Base level is DEBUG

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler for all logs
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        if name != 'metrics': # Avoid double logging in the same file
            # Error file handler
            eh = logging.FileHandler(self.config.ERROR_LOG_FILE)
            eh.setLevel(logging.ERROR)
            eh.setFormatter(formatter)
            logger.addHandler(eh)

        return logger

logger_manager = LoggerManager(config)
logger = logger_manager.logger
metrics_logger = logger_manager.metrics_logger
logger.info("Logging system initialized.")

# --- Custom Exception with Enhanced Context ---
class DropboxOperationError(Exception):
    """Custom exception for Dropbox related operations."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.original_exception = original_exception
        self.details = details

    def __str__(self):
        base_message = super().__str__()
        details_str = f" Details: {self.details}" if self.details else ""
        original_exc_str = f" Original Exception: {repr(self.original_exception)}" if self.original_exception else ""
        return f"{base_message}{details_str}{original_exc_str}"

# --- Authentication with Retry Mechanism and Detailed Logging ---
class DropboxAuthenticator:
    def __init__(self, app_key: str, app_secret: str, token_file: str, access_token: Optional[str] = None):
        self.app_key = app_key
        self.app_secret = app_secret
        self.token_file = token_file
        self.access_token = access_token
        self.lock = threading.Lock()

    def load_token(self) -> Optional[str]:
        """Loads the access token from file with detailed logging."""
        if self.access_token:
            logger.info("Using access token from environment variable.")
            return self.access_token
        with self.lock:
            try:
                logger.debug(f"Attempting to load token from {self.token_file}")
                with open(self.token_file, "r", encoding="utf-8") as f:
                    token_data = json.load(f)
                    logger.info(f"Token loaded successfully from {self.token_file}")
                    return token_data.get('access_token')
            except FileNotFoundError:
                logger.warning(f"Token file not found: {self.token_file}")
                return None
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding token file {self.token_file}: {e}", exc_info=True)
                return None
            except Exception as e:
                logger.error(f"Unexpected error loading token from {self.token_file}: {e}", exc_info=True)
                return None

    def save_token(self, access_token: str):
        """Saves the access token to file with detailed logging and error handling."""
        if self.access_token:
            logger.info("Token was loaded from environment, not saving to file.")
            return
        with self.lock:
            try:
                logger.debug(f"Attempting to save token to {self.token_file}")
                with open(self.token_file, "w", encoding="utf-8") as f:
                    json.dump({'access_token': access_token}, f, indent=4)
                logger.info(f"Authentication successful! Token saved to {self.token_file}")
            except IOError as e:
                logger.error(f"IOError saving token to {self.token_file}: {e}", exc_info=True)
                raise DropboxOperationError(f"Failed to save Dropbox token to {self.token_file}: {e}", e)
            except TypeError as e:
                logger.error(f"TypeError saving token to {self.token_file}: {e}", exc_info=True)
                raise DropboxOperationError(f"Failed to save Dropbox token due to type error: {e}", e)
            except Exception as e:
                logger.error(f"Unexpected error saving token to {self.token_file}: {e}", exc_info=True)
                raise DropboxOperationError(f"Failed to save Dropbox token due to unexpected error: {e}", e)

    def authenticate(self) -> dropbox.Dropbox:
        """Authenticates with Dropbox using OAuth2 with robust error handling and token management."""
        logger.info("Starting Dropbox authentication process.")
        token = self.load_token()
        if token:
            logger.info("Attempting authentication with existing token.")
            try:
                dbx = dropbox.Dropbox(token)
                dbx.users_get_current_account()  # Verify token
                logger.info("Successfully authenticated using existing token.")
                return dbx
            except AuthError as e:
                logger.warning(f"Existing token is invalid: {e}. Initiating new authentication flow.")
            except Exception as e:
                logger.error(f"Error verifying existing token: {e}", exc_info=True)

        # OAuth2 flow for new authentication
        logger.info("Initiating new authentication flow.")
        auth_flow = DropboxOAuth2FlowNoRedirect(self.app_key, self.app_secret)
        authorize_url = auth_flow.start()
        print(Fore.BLUE + "1. Go to this URL and log in:" + Style.RESET_ALL, authorize_url)
        print(Fore.BLUE + "2. Copy the authorization code and paste it below." + Style.RESET_ALL)
        auth_code = input("Enter the authorization code here: ").strip()

        try:
            logger.debug("Attempting to finish authentication with authorization code.")
            oauth_result = auth_flow.finish(auth_code)
            self.save_token(oauth_result.access_token)
            logger.info("New access token obtained and saved.")
            return dropbox.Dropbox(oauth_result.access_token)
        except Exception as e:
            logger.error(f"Error during authentication: {e}", exc_info=True)
            raise DropboxOperationError(f"Dropbox authentication failed: {e}", e)

# --- Enhanced Folder Listing with Real-time Reporting, Metrics, and Cancellation Support ---
class DropboxFolderLister:
    def __init__(self, dbx: dropbox.Dropbox, max_depth: int):
        self.dbx = dbx
        self.max_depth = max_depth
        self.output_data: Dict[str, Any] = {"folders": [], "files": []}
        self.metrics: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "total_files": 0,
            "total_folders": 0,
            "api_errors": 0,
            "file_errors": 0,
            "total_size_bytes": 0,
            "max_depth_reached": max_depth,
            "api_call_details": defaultdict(int)
        }
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        self.folder_queue = deque([("", 0)])  # Start with the root folder
        self.file_type_counts = defaultdict(int)
        self.processed_files_count = 0
        self.processed_folders_count = 0
        self.last_report_time = time.time()
        self.start_time = time.time()
        self._shutdown_event = threading.Event()
        self.lock = threading.Lock()

    def log_metric(self, name: str, value: Any):
        """Logs a metric with the metrics logger."""
        metrics_logger.info(f"{name}: {value}")

    def increment_metric(self, name: str):
        """Increments a metric."""
        with self.lock:
            self.metrics[name] = self.metrics.get(name, 0) + 1

    def record_file_size(self, size: int):
        """Records the size of a processed file."""
        with self.lock:
            self.metrics["total_size_bytes"] += size

    def _execute_api_call(self, func, *args, **kwargs):
        retries = 0
        while retries <= config.API_MAX_RETRIES:
            if self._shutdown_event.is_set():
                logger.info("API call execution cancelled due to shutdown signal.")
                raise InterruptedError("Operation cancelled.")
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                api_name = func.__name__
                with self.lock:
                    self.metrics["api_call_details"][api_name] += 1
                logger.debug(f"API call '{api_name}' executed successfully in {duration:.4f} seconds.")
                return result
            except RateLimitError as e:
                self.increment_metric("api_errors")
                retry_delay = e.backoff if e.backoff else config.API_RETRY_DELAY
                logger.warning(f"Rate limit exceeded for API call '{func.__name__}'. Retrying in {retry_delay} seconds (attempt {retries + 1}/{config.API_MAX_RETRIES}).")
                time.sleep(retry_delay)
                retries += 1
            except ApiError as e:
                self.increment_metric("api_errors")
                logger.error(f"API error in '{func.__name__}': {e}", exc_info=True, extra={'details': {'error_tag': e.error.get_tag() if hasattr(e.error, 'get_tag') else None}})
                raise DropboxOperationError(f"Dropbox API error in '{func.__name__}'", original_exception=e, details={'error_tag': e.error.get_tag() if hasattr(e.error, 'get_tag') else None})
            except Exception as e:
                self.increment_metric("api_errors")
                logger.error(f"Unexpected error in API call '{func.__name__}': {e}", exc_info=True)
                raise DropboxOperationError(f"Unexpected error during Dropbox API call '{func.__name__}'", original_exception=e)
        raise DropboxOperationError(f"API call '{func.__name__}' failed after {config.API_MAX_RETRIES} retries.")

    def list_folder_chunked(self, path: str) -> List[dropbox.files.Metadata]:
        """Lists a folder's contents in chunks with detailed error handling and logging."""
        entries: List[dropbox.files.Metadata] = []
        cursor = None
        logger.info(f"Starting chunked listing for folder: {path}")
        while True:
            if self._shutdown_event.is_set():
                logger.info(f"Chunked listing for '{path}' cancelled due to shutdown signal.")
                break
            try:
                if cursor:
                    logger.debug(f"Listing folder chunk - continuing from cursor.")
                    result = self._execute_api_call(self.dbx.files_list_folder_continue, cursor)
                else:
                    logger.debug(f"Listing folder chunk: {path}")
                    result = self._execute_api_call(self.dbx.files_list_folder, path, limit=config.LIST_FOLDER_CHUNK_SIZE)

                entries.extend(result.entries)
                if not result.has_more:
                    break
                cursor = result.cursor
                logger.debug(f"Received {len(result.entries)} entries for '{path}'. Has more: {result.has_more}")
            except DropboxOperationError as e:
                logger.error(f"Error listing folder '{path}': {e}", exc_info=True)
                raise
            except InterruptedError:
                break
            except Exception as e:
                logger.error(f"Unexpected error listing folder '{path}': {e}", exc_info=True)
                raise DropboxOperationError(f"Unexpected error listing folder '{path}': {e}", e)
        logger.info(f"Successfully listed folder: {path}, found {len(entries)} entries.")
        return entries

    def get_detailed_file_info(self, entry: dropbox.files.FileMetadata) -> Optional[Dict[str, Any]]:
        """Fetches detailed metadata for a file with error handling."""
        try:
            logger.debug(f"Fetching detailed info for file: {entry.path_lower}")
            metadata = self._execute_api_call(self.dbx.files_get_metadata, entry.path_lower)
            if isinstance(metadata, dropbox.files.FileMetadata):
                file_info = {
                    "name": metadata.name,
                    "path_lower": metadata.path_lower,
                    "size": metadata.size,
                    "modified": metadata.server_modified.isoformat(),
                    "client_modified": metadata.client_modified.isoformat(),
                    "content_hash": metadata.content_hash
                }
                logger.debug(f"Detailed info fetched for file: {entry.path_lower}")
                return file_info
            else:
                logger.warning(f"Expected FileMetadata, got: {type(metadata)} for {entry.path_lower}")
                return None
        except DropboxOperationError as e:
            logger.error(f"API error getting metadata for '{entry.path_lower}': {e}", exc_info=True)
            return None
        except InterruptedError:
            return None
        except Exception as e:
            logger.error(f"Error getting detailed info for file '{entry.name}': {e}", exc_info=True)
            self.increment_metric("file_errors")
            return None

    def process_entry(self, entry: dropbox.files.Metadata, depth: int):
        """Processes a single entry (file or folder) and extracts detailed information."""
        try:
            if isinstance(entry, dropbox.files.FolderMetadata):
                logger.debug(f"Processing folder: {entry.name}, depth: {depth}")
                self.increment_metric("total_folders")
                with self.lock:
                    self.output_data["folders"].append({
                        "name": entry.name,
                        "path_lower": entry.path_lower,
                        "depth": depth
                    })
                self.processed_folders_count += 1
                return True
            elif isinstance(entry, dropbox.files.FileMetadata):
                logger.debug(f"Processing file: {entry.name}, depth: {depth}")
                self.increment_metric("total_files")
                self.record_file_size(entry.size)
                extension = os.path.splitext(entry.name)[1].lower() or 'no_extension'
                with self.lock:
                    self.file_type_counts[extension] += 1
                file_info = self.get_detailed_file_info(entry)
                if file_info:
                    with self.lock:
                        self.output_data["files"].append(file_info)
                self.processed_files_count += 1
                return False
            else:
                logger.warning(f"Unknown entry type: {type(entry)} - {entry.name}")
                return False
        except Exception as e:
            logger.error(f"Error processing entry '{entry.name}': {e}", exc_info=True)
            self.increment_metric("file_errors")
            return False

    def process_folder(self, path: str, depth: int):
        """Processes a folder, lists its contents, and adds subfolders to the queue."""
        if depth >= self.max_depth:
            logger.debug(f"Max depth reached for path: {path}")
            return

        try:
            logger.info(f"Processing folder: {path} at depth: {depth}")
            entries = self.list_folder_chunked(path)
            for entry in entries:
                if self._shutdown_event.is_set():
                    logger.info(f"Processing of '{path}' interrupted.")
                    break
                self.process_entry(entry, depth)
                if isinstance(entry, dropbox.files.FolderMetadata) and depth + 1 <= self.max_depth:
                    self.folder_queue.append((entry.path_lower, depth + 1))
        except DropboxOperationError as e:
            logger.error(f"Error processing path '{path}': {e}", exc_info=True)
        except InterruptedError:
            logger.info(f"Processing of folder '{path}' was interrupted.")
        except Exception as e:
            logger.error(f"Unexpected error during folder processing '{path}': {e}", exc_info=True)

    def list_folder_tree_concurrent(self):
        """Lists the folder tree concurrently using a thread pool and a queue."""
        self.metrics["start_time"] = datetime.now().isoformat()
        self.log_metric("Operation Start Time", self.metrics["start_time"])
        logger.info("Starting concurrent folder listing.")

        with self.executor as executor:
            futures = {executor.submit(self.process_folder, path, depth): (path, depth) for path, depth in [self.folder_queue.popleft()]}
            while futures:
                done, not_done = [], {}
                for future in as_completed(futures, timeout=config.REPORT_INTERVAL):
                    try:
                        future.result()
                        done.append(future)
                    except DropboxOperationError as e:
                        logger.error(f"Error processing folder {futures[future][0]}: {e}")
                        done.append(future) # Mark as done even with exception
                    except InterruptedError:
                        logger.info(f"Folder processing for {futures[future][0]} was interrupted.")
                        done.append(future)
                    except Exception as e:
                        logger.error(f"Unexpected error processing folder {futures[future][0]}: {e}", exc_info=True)
                        done.append(future)

                for future in done:
                    del futures[future]

                self._report_progress()
                while self.folder_queue and len(futures) < config.MAX_WORKERS and not self._shutdown_event.is_set():
                    path, depth = self.folder_queue.popleft()
                    futures[executor.submit(self.process_folder, path, depth)] = (path, depth)

                if self._shutdown_event.is_set():
                    logger.info("Shutdown signal received, stopping new folder submissions.")
                    break

        logger.info("Waiting for all pending tasks to complete.")
        self.executor.shutdown(wait=True)
        self.metrics["end_time"] = datetime.now().isoformat()
        self.metrics["duration_seconds"] = time.time() - self.start_time
        self.log_metric("Operation End Time", self.metrics["end_time"])
        self.log_metric("Total Duration (seconds)", f"{self.metrics['duration_seconds']:.2f}")
        self.log_metric("Total API Calls", sum(self.metrics['api_call_details'].values()))
        logger.info("Concurrent folder listing completed.")

    def _report_progress(self):
        """Reports the progress of the folder listing."""
        current_time = time.time()
        if current_time - self.last_report_time >= config.REPORT_INTERVAL:
            elapsed_time = current_time - self.start_time
            files_per_second = self.processed_files_count / elapsed_time if elapsed_time > 0 else 0
            folders_per_second = self.processed_folders_count / elapsed_time if elapsed_time > 0 else 0
            api_errors = self.metrics.get('api_errors', 0)
            file_errors = self.metrics.get('file_errors', 0)
            total_entries = self.processed_files_count + self.processed_folders_count
            logger.info(
                f"Progress Update - Elapsed Time: {elapsed_time:.2f}s, "
                f"Folders: {self.processed_folders_count} ({folders_per_second:.2f}/s), "
                f"Files: {self.processed_files_count} ({files_per_second:.2f}/s), "
                f"Total Entries: {total_entries}, "
                f"API Errors: {api_errors}, File Errors: {file_errors}"
            )
            metrics_logger.info(
                f"Progress Metrics - Folders: {self.processed_folders_count}, Files: {self.processed_files_count}, "
                f"API Errors: {api_errors}, File Errors: {file_errors}, Total Entries: {total_entries}"
            )
            self.last_report_time = current_time

    def get_metrics(self) -> Dict[str, Any]:
        """Returns comprehensive metrics about the folder listing process."""
        return self.metrics

    def save_output(self, filename: str = config.OUTPUT_FILE):
        """Saves the folder structure output to a file in JSON format with error handling."""
        try:
            logger.info(f"Saving folder structure to: {filename}")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.output_data, f, indent=4)
            logger.info(f"Folder structure saved successfully to: {filename}")
        except IOError as e:
            logger.error(f"IOError saving output to {filename}: {e}", exc_info=True)
            raise DropboxOperationError(f"Error saving output to {filename}: {e}", e)
        except TypeError as e:
            logger.error(f"TypeError saving output to {filename}: {e}", exc_info=True)
            raise DropboxOperationError(f"Type error saving output to {filename}: {e}", e)
        except Exception as e:
            logger.error(f"Unexpected error saving output to {filename}: {e}", exc_info=True)
            raise DropboxOperationError(f"Unexpected error saving output to {filename}: {e}", e)

    def save_metrics(self, filename: str = config.METRICS_FILE):
        """Saves the metrics to a JSON file with error handling."""
        try:
            logger.info(f"Saving metrics to: {filename}")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.metrics, f, indent=4, default=str)
            logger.info(f"Metrics saved successfully to: {filename}")
            for key, value in self.metrics.items():
                self.log_metric(f"Final Metric - {key}", value)
        except IOError as e:
            logger.error(f"IOError saving metrics to {filename}: {e}", exc_info=True)
            raise DropboxOperationError(f"Error saving metrics to {filename}: {e}", e)
        except TypeError as e:
            logger.error(f"TypeError saving metrics to {filename}: {e}", exc_info=True)
            raise DropboxOperationError(f"Type error saving metrics to {filename}: {e}", e)
        except Exception as e:
            logger.error(f"Unexpected error saving metrics to {filename}: {e}", exc_info=True)
            raise DropboxOperationError(f"Unexpected error saving metrics to {filename}: {e}", e)

    def stop(self):
        """Initiates a graceful shutdown of the folder listing process."""
        logger.info("Received shutdown signal. Initiating graceful shutdown...")
        self._shutdown_event.set()
        # Optionally, you might want to wait for the threads to finish here,
        # but in this implementation, the threads will check the event and exit.

if __name__ == "__main__":
    authenticator = DropboxAuthenticator(config.APP_KEY, config.APP_SECRET, config.TOKEN_FILE, config.ACCESS_TOKEN)
    try:
        dbx = authenticator.authenticate()
        lister = DropboxFolderLister(dbx, max_depth=config.MAX_DEPTH)

        print("\n" + Fore.GREEN + "Starting Dropbox Folder Structure listing:" + Style.RESET_ALL)
        start_time = time.time()
        thread = threading.Thread(target=lister.list_folder_tree_concurrent)
        thread.start()

        try:
            while thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print(Fore.YELLOW + "\nKeyboard interrupt received. Initiating shutdown..." + Style.RESET_ALL)
            logger.info("Keyboard interrupt received. Initiating shutdown...")
            lister.stop()
            thread.join(timeout=10) # Wait for the thread to finish
            if thread.is_alive():
                logger.warning("Folder listing thread did not terminate gracefully.")
            else:
                logger.info("Folder listing thread terminated.")

        end_time = time.time()
        print(Fore.GREEN + "Dropbox Folder Structure listing completed." + Style.RESET_ALL)

        if config.OUTPUT_FORMAT == 'text':
            print("\n" + Fore.YELLOW + "Iconified Colourised Directory Tree:" + Style.RESET_ALL + "\n")
            sorted_folders = sorted(lister.output_data['folders'], key=lambda f: f['depth'])
            for depth, folders in groupby(sorted_folders, key=lambda f: f['depth']):
                for folder in folders:
                    indent = "    " * depth
                    print(f"{indent}{Fore.CYAN}📁 {folder['name']}{Style.RESET_ALL}")

            print("\n" + Fore.YELLOW + "File Type Summary:" + Style.RESET_ALL)
            for ext, count in lister.file_type_counts.items():
                print(f"  {Fore.MAGENTA}{ext}:{Style.RESET_ALL} {count}")

        elif config.OUTPUT_FORMAT == 'json':
            print("\n" + Fore.YELLOW + "Dropbox Folder Structure (JSON):" + Style.RESET_ALL + "\n")
            print(json.dumps(lister.output_data, indent=4))

        lister.save_output()
        metrics = lister.get_metrics()
        lister.save_metrics()

        logger.info(f"Total files processed: {metrics.get('total_files', 0)}")
        logger.info(f"Total folders processed: {metrics.get('total_folders', 0)}")
        logger.info(f"Time taken: {end_time - start_time:.2f} seconds")

    except DropboxOperationError as e:
        logger.critical(f"A critical Dropbox error occurred: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
