import hashlib
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
import aiofiles
import json
import logging
from tkinter import filedialog, Tk, ttk
from typing import Dict, List, Tuple, Union, Callable, Coroutine, Any
from functools import wraps
import os

# Configure logging to the highest standards
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Convert string paths to Path objects using os.path for compatibility
SOURCE_DIR: Path = Path(os.path.expanduser("~/home/lloyd/Dropbox/To_Sort"))
DUPLICATES_DIR: Path = Path(os.path.expanduser("~/home/lloyd/duplicates"))
SORTED_DIR: Path = Path(os.path.expanduser("~/home/lloyd/sorted"))
ERRORS_DIR: Path = Path(os.path.expanduser("~/home/lloyd/errors"))
INDEX_FILE: Path = Path(os.path.expanduser("~/home/lloyd/index.json"))
ERROR_INDEX_FILE: Path = Path(os.path.expanduser("~/home/lloyd/error_index.json"))

# Ensure directories exist with detailed logging
for directory in [DUPLICATES_DIR, SORTED_DIR, ERRORS_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured existence of directory: {directory}")
    except Exception as e:
        logger.error(f"Failed to ensure directory {directory}: {e}", exc_info=True)


# GUI Components with extensive documentation and type annotations
class FileBrowserGUI:
    """
    A sophisticated file browsing GUI facilitating directory selection for file processing.
    Implements forward and back navigation alongside a dynamic dropdown menu for file tree display.
    """

    def __init__(self, root: Tk) -> None:
        self.root: Tk = root
        self.root.title("File Browser")
        self.current_path: Path = SOURCE_DIR
        self.history: List[Path] = [self.current_path]
        self.history_index: int = 0

        # Navigation Frame
        self.nav_frame: ttk.Frame = ttk.Frame(self.root)
        self.nav_frame.pack(fill="x", padx=5, pady=5)

        # Back Button
        self.back_button: ttk.Button = ttk.Button(
            self.nav_frame, text="Back", command=self.go_back
        )
        self.back_button.pack(side="left", padx=5)

        # Forward Button
        self.forward_button: ttk.Button = ttk.Button(
            self.nav_frame, text="Forward", command=self.go_forward
        )
        self.forward_button.pack(side="left", padx=5)

        # Path Dropdown
        self.path_var: ttk.Combobox = ttk.Combobox(self.nav_frame, width=60)
        self.path_var.pack(fill="x", expand=True)
        self.path_var.bind("<<ComboboxSelected>>", self.path_selected)

        self.update_path_dropdown()

    def update_path_dropdown(self) -> None:
        """
        Dynamically updates the dropdown menu with the current directory's subdirectories and files.
        """
        paths: List[str] = [str(p) for p in self.current_path.iterdir()]
        self.path_var["values"] = paths
        if paths:
            self.path_var.current(0)

    def path_selected(self, event: Any) -> None:
        """
        Handles path selection from the dropdown menu, navigating to the selected directory.
        """
        selected_path: Path = Path(self.path_var.get())
        if selected_path.is_dir():
            self.navigate_to(selected_path)

    def navigate_to(self, path: Path) -> None:
        """
        Navigates to the specified directory, updating navigation history.
        """
        self.current_path = path
        self.history = self.history[: self.history_index + 1] + [path]
        self.history_index += 1
        self.update_path_dropdown()

    def go_back(self) -> None:
        """
        Navigates to the previous directory in the navigation history.
        """
        if self.history_index > 0:
            self.history_index -= 1
            self.current_path = self.history[self.history_index]
            self.update_path_dropdown()

    def go_forward(self) -> None:
        """
        Navigates to the next directory in the navigation history, if available.
        """
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_path = self.history[self.history_index]
            self.update_path_dropdown()


def launch_file_browser() -> Path:
    """
    Launches the file browser GUI, returning the selected directory path.
    """
    root: Tk = Tk()
    gui: FileBrowserGUI = FileBrowserGUI(root)
    root.mainloop()
    return gui.current_path


# Decorators with detailed documentation and type annotations
def async_error_handler(
    func: Callable[..., Coroutine[Any, Any, Any]]
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """
    A decorator to handle exceptions in asynchronous functions, log them, and provide user feedback.
    Extends functionality to ensure robust error handling and logging.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"An error occurred in {func.__name__}: {e}", exc_info=True)
            if not ERRORS_DIR.exists():
                ERRORS_DIR.mkdir(parents=True, exist_ok=True)
            error_file: Path = (
                ERRORS_DIR
                / f"{func.__name__}_error_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
            )
            async with aiofiles.open(error_file, "w") as f:
                await f.write(str(e))
            raise e

    return wrapper


# Utility Functions with comprehensive type annotations and detailed documentation
async def calculate_file_hash(file_path: Path) -> str:
    """
    Asynchronously calculates the SHA-256 hash of a file, reading in 64kb chunks.
    """
    BUF_SIZE: int = 65536
    sha256 = hashlib.sha256()  # Removed explicit type hint
    async with aiofiles.open(file_path, "rb") as f:
        while True:
            data: bytes = await f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


async def move_file(file_path: Path, destination: Path) -> None:
    """
    Moves a file to a specified destination directory, ensuring the destination exists.
    """
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)
    destination_file: Path = destination / file_path.name
    shutil.move(str(file_path), str(destination_file))


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
    """
    file_extension: str = file_path.suffix
    destination_dir: Path = SORTED_DIR / file_extension.strip(".")
    today_date: str = datetime.now().strftime("_%d%m%y")
    standardized_name: str = f"{file_hash}{today_date}{file_extension}"
    destination_file: Path = destination_dir / standardized_name

    if file_hash in duplicates_index:
        await move_file(file_path, DUPLICATES_DIR)
        duplicates_index[file_hash].append(str(file_path))
    else:
        duplicates_index[file_hash] = [str(file_path)]
        kept_files_index[str(destination_file)] = str(file_path)
        if not destination_dir.exists():
            destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(destination_file))


async def process_directory(
    directory: Path,
    duplicates_index: Dict[str, List[str]],
    kept_files_index: Dict[str, str],
) -> None:
    """
    Recursively processes each file in a directory and its subdirectories.
    Employs advanced asynchronous programming techniques for efficiency.
    """
    for item in directory.iterdir():
        if item.is_file():
            file_hash: str = await calculate_file_hash(item)
            await process_file(item, file_hash, duplicates_index, kept_files_index)
        elif item.is_dir():
            await process_directory(item, duplicates_index, kept_files_index)


@async_error_handler
async def main() -> None:
    """
    The main function orchestrating the file processing workflow.
    Implements a sophisticated GUI for directory selection and processes files accordingly.
    """
    selected_directory: Path = launch_file_browser()
    duplicates_index: Dict[str, List[str]] = {}
    kept_files_index: Dict[str, str] = {}
    await process_directory(selected_directory, duplicates_index, kept_files_index)
    async with aiofiles.open(INDEX_FILE, "w") as f:
        await f.write(json.dumps(kept_files_index, indent=4))
    async with aiofiles.open(ERROR_INDEX_FILE, "w") as f:
        await f.write(json.dumps(duplicates_index, indent=4))


if __name__ == "__main__":
    asyncio.run(main())
