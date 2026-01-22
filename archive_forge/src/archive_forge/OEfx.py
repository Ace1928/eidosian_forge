#!/home/lloyd/indego/bin/python

import hashlib
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
import aiofiles
import logging
from typing import Dict, List, Tuple, Union, Callable, Coroutine, Any
from functools import wraps
import threading
import ctypes
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QTreeView,
    QFileSystemModel,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)
from PyQt5.QtCore import QDir
from PyQt5.QtCore import Qt
import os
import ctypes

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants for file paths
SOURCE_DIR = Path("/home/lloyd/Dropbox/To_Sort")
DUPLICATES_DIR = Path("/home/lloyd/duplicates")
SORTED_DIR = Path("/home/lloyd/sorted")
ERRORS_DIR = Path("/home/lloyd/errors")
INDEX_FILE = Path("/home/lloyd/index.json")
ERROR_INDEX_FILE = Path("/home/lloyd/error_index.json")

# Ensure directories exist
for directory in [DUPLICATES_DIR, SORTED_DIR, ERRORS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Ensure directories exist
for directory in [DUPLICATES_DIR, SORTED_DIR, ERRORS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Async error handler decorator
def async_error_handler(
    func: Callable[..., Coroutine[Any, Any, Any]]
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """
    A decorator that wraps an asynchronous function to handle and log any exceptions that occur during execution.

    Args:
        func (Callable[..., Coroutine[Any, Any, Any]]): The asynchronous function to be wrapped.

    Returns:
        Callable[..., Coroutine[Any, Any, Any]]: The wrapped asynchronous function with error handling.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        The wrapper function that executes the wrapped asynchronous function and handles any exceptions.

        Args:
            *args (Any): Positional arguments to be passed to the wrapped function.
            **kwargs (Any): Keyword arguments to be passed to the wrapped function.

        Returns:
            Any: The result of the wrapped function if no exceptions occur.

        Raises:
            Exception: If an exception occurs during the execution of the wrapped function.
        """
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"An error occurred in {func.__name__}: {e}", exc_info=True)
            if not ERRORS_DIR.exists():
                ERRORS_DIR.mkdir(parents=True, exist_ok=True)
            error_file = (
                ERRORS_DIR
                / f"{func.__name__}_error_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
            )
            async with aiofiles.open(error_file, "a") as f:
                await f.write(str(e))
            raise

    return wrapper


# File Browser GUI
class FileBrowserWidget(Optional[QWidget, None]):
    """
    An advanced file browser widget with a tree view, progress bar, and interactive navigation.
    """

    def __init__(self, parent: QWidget) -> None:
        """
        Initializes the FileBrowserWidget instance.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("File Browser")
        self.resize(800, 600)

        # Create file system model
        self.model = QFileSystemModel()
        self.model.setRootPath(str(SOURCE_DIR))

        # Create tree view
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.tree_view.setRootIndex(self.model.index(str(SOURCE_DIR)))
        self.tree_view.setColumnWidth(0, 250)
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setAnimated(True)
        self.tree_view.setExpandsOnDoubleClick(True)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.setWordWrap(True)
        self.tree_view.setHeaderHidden(False)
        self.tree_view.setIndentation(20)
        self.tree_view.setSelectionMode(QTreeView.ExtendedSelection)
        self.tree_view.setEditTriggers(QTreeView.NoEditTriggers)
        self.tree_view.setDragEnabled(True)
        self.tree_view.setDragDropMode(QTreeView.DragOnly)
        self.tree_view.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.tree_view.setDropIndicatorShown(True)
        self.tree_view.setUniformRowHeights(True)

        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Processing files: %p%")

        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.tree_view)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def update_progress(self, value: int) -> None:
        """
        Updates the progress bar value.

        Args:
            value (int): The progress value to set.
        """
        self.progress_bar.setValue(value)


def launch_file_browser() -> Path:
    """
    Launches the file browser GUI, returning the selected directory path.

    Returns:
        Path: The selected directory path from the file browser GUI.
    """
    app = QApplication(sys.argv)
    file_browser = FileBrowserWidget(None)
    file_browser.show()
    app.exec_()
    selected_indexes = file_browser.tree_view.selectedIndexes()
    if selected_indexes:
        selected_path = Path(file_browser.model.filePath(selected_indexes[0]))
        return selected_path
    return SOURCE_DIR


# Utility Functions with comprehensive type annotations and detailed documentation for asynchronous file processing
async def calculate_file_hash(file_path: Path) -> str:
    """
    Asynchronously calculates the SHA-256 hash of a file, reading in 64kb chunks.

    Args:
        file_path (Path): The path to the file for which the hash needs to be calculated.

    Returns:
        str: The hexadecimal digest of the calculated SHA-256 hash.
    """
    BUF_SIZE: int = 65536  # Define buffer size for efficient reading
    sha256 = hashlib.sha256()  # Initialize SHA-256 hash object
    async with aiofiles.open(
        file_path, "rb"
    ) as f:  # Asynchronously open file for reading in binary mode
        while True:
            data: bytes = await f.read(BUF_SIZE)  # Asynchronously read a chunk of data
            if not data:
                break  # Exit loop if end of file is reached
            sha256.update(data)  # Update hash object with chunk
    return sha256.hexdigest()  # Return hexadecimal digest of the hash


async def move_file(file_path: Path, destination: Path) -> None:
    """
    Moves a file to a specified destination directory, ensuring the destination exists.

    Args:
        file_path (Path): The path to the file to be moved.
        destination (Path): The destination directory path where the file should be moved.
    """
    if not destination.exists():
        destination.mkdir(
            parents=True, exist_ok=True
        )  # Ensure destination directory exists
    destination_file: Path = (
        destination / file_path.name
    )  # Define destination file path
    shutil.move(str(file_path), str(destination_file))  # Move file to destination


@async_error_handler
async def process_file(
    file_path: Path,
    file_hash: str,
    duplicates_index: Dict[str, List[str]],
    kept_files_index: Dict[str, str],
) -> None:
    """
    Processes a file by checking for duplicates and moving it to the appropriate directory.
    Implements advanced hashing and file management strategies.

    Args:
        file_path (Path): The path to the file to be processed.
        file_hash (str): The calculated hash of the file.
        duplicates_index (Dict[str, List[str]]): A dictionary to store duplicate file information.
        kept_files_index (Dict[str, str]): A dictionary to store information about kept files.
    """
    file_extension: str = file_path.suffix  # Extract file extension
    destination_dir: Path = SORTED_DIR / file_extension.strip(
        "."
    )  # Define destination directory based on file extension
    today_date: str = datetime.now().strftime(
        "_%d%m%y"
    )  # Get today's date in specified format
    standardized_name: str = (
        f"{file_hash}{today_date}{file_extension}"  # Construct standardized file name
    )
    destination_file: Path = (
        destination_dir / standardized_name
    )  # Define destination file path

    if file_hash in duplicates_index:
        await move_file(file_path, DUPLICATES_DIR)  # Move file to duplicates directory
        duplicates_index[file_hash].append(
            str(file_path)
        )  # Add file path to duplicates index
    else:
        duplicates_index[file_hash] = [
            str(file_path)
        ]  # Initialize list with file path in duplicates index
        kept_files_index[str(destination_file)] = str(
            file_path
        )  # Add file path to kept files index
        if not destination_dir.exists():
            destination_dir.mkdir(
                parents=True, exist_ok=True
            )  # Ensure destination directory exists
        shutil.move(str(file_path), str(destination_file))  # Move file to destination


async def process_directory(
    directory: Path,
    duplicates_index: Dict[str, List[str]],
    kept_files_index: Dict[str, str],
) -> None:
    """
    Recursively processes each file in a directory and its subdirectories.
    Employs advanced asynchronous programming techniques for efficiency.

    Args:
                directory (Path): The path to the directory to be processed.
        duplicates_index (Dict[str, List[str]]): A dictionary to store duplicate file information.
        kept_files_index (Dict[str, str]): A dictionary to store information about kept files.
    """
    for item in directory.iterdir():  # Iterate over items in directory
        if item.is_file():  # Check if item is a file
            file_hash: str = await calculate_file_hash(item)  # Calculate file hash
            await process_file(
                item, file_hash, duplicates_index, kept_files_index
            )  # Process file
        elif item.is_dir():  # Check if item is a directory
            await process_directory(
                item, duplicates_index, kept_files_index
            )  # Recursively process directory


def run_asyncio_loop(loop: asyncio.AbstractEventLoop, coro: Coroutine) -> None:
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)


async def async_main() -> None:
    """
    The main asynchronous function to be executed.
    """
    selected_directory = launch_file_browser()
    duplicates_index: Dict[str, List[str]] = {}
    kept_files_index: Dict[str, str] = {}
    await process_directory(selected_directory, duplicates_index, kept_files_index)


def is_admin() -> bool:
    """
    Determines if the script is running with administrator or superuser privileges across different operating systems,
    including Windows, Linux, and Android. It ensures that on Android, the script does not run as admin unless the user
    has superuser privileges, which is generally not common. If not running with the required privileges, it prompts the
    user to relaunch the program with elevated privileges.

    Returns:
        bool: True if the script has administrator privileges or is running as root/superuser, False otherwise.
    """
    try:
        if os.name == "nt":  # Checks if the operating system is Windows
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            if not is_admin:
                logging.info("Requesting elevated privileges on Windows.")
                response = input(
                    "This script requires administrator privileges. Would you like to relaunch with elevated privileges? (y/n): "
                )
                if response.lower() == "y":
                    ctypes.windll.shell32.ShellExecuteW(
                        None, "runas", sys.executable, " ".join(sys.argv), None, 1
                    )
                    sys.exit(0)
            return is_admin
        elif (
            os.name == "posix"
        ):  # Checks if the operating system is POSIX-compliant (Linux, Unix, macOS, etc.)
            is_root = os.geteuid() == 0
            if not is_root:
                logging.info("Requesting superuser privileges on POSIX systems.")
                response = input(
                    "This script requires superuser privileges. Would you like to attempt to relaunch with sudo? (y/n): "
                )
                if response.lower() == "y":
                    os.execvp("sudo", ["sudo"] + sys.argv)
            return is_root
        elif os.name == "android":  # Specific check for Android operating system
            # Android does not typically allow applications to run with superuser privileges unless rooted
            # This check attempts to run a command that requires superuser privileges and checks the outcome
            is_superuser = os.system("su -c id") == 0
            if not is_superuser:
                logging.info(
                    "Superuser privileges are required on Android, but not available."
                )
            return is_superuser
        else:
            # For any other operating systems not explicitly checked, direct to support for upgrade/implementation
            logging.info(
                "Operating system not currently supported. Contact support for implementation."
            )
            return False
    except Exception as e:
        # Log any exceptions that occur during the check for administrative privileges
        logging.error(f"Error checking admin privileges: {e}")
        return False


def main() -> None:
    """
    The main entry point of the script.
    """
    if not is_admin():
        sys.exit(0)

    loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
    t: threading.Thread = threading.Thread(
        target=run_asyncio_loop, args=(loop, async_main())
    )
    t.start()
    t.join()


if __name__ == "__main__":
    main()
