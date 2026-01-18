import importlib
import inspect
from functools import wraps
from typing import TYPE_CHECKING, Optional
from .download.download_config import DownloadConfig
from .download.streaming_download_manager import (
from .utils.logging import get_logger
from .utils.patching import patch_submodule
from .utils.py_utils import get_imports
def wrap_auth(function):

    @wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, download_config=download_config, **kwargs)
    wrapper._decorator_name_ = 'wrap_auth'
    return wrapper