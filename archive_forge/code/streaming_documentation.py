import importlib
import inspect
from functools import wraps
from typing import TYPE_CHECKING, Optional
from .download.download_config import DownloadConfig
from .download.streaming_download_manager import (
from .utils.logging import get_logger
from .utils.patching import patch_submodule
from .utils.py_utils import get_imports
Extend the dataset builder module and the modules imported by it to support streaming.

    Args:
        builder (:class:`DatasetBuilder`): Dataset builder instance.
    