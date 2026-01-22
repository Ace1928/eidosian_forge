import importlib
import shutil
import threading
import warnings
from typing import List
import fsspec
import fsspec.asyn
from fsspec.implementations.local import LocalFileSystem
from ..utils.deprecation_utils import deprecated
from . import compression
def _reset_fsspec_lock() -> None:
    """
    Clear reference to the loop and thread.
    This is necessary otherwise HTTPFileSystem hangs in the ML training loop.
    Only required for fsspec >= 0.9.0
    See https://github.com/fsspec/gcsfs/issues/379
    """
    if hasattr(fsspec.asyn, 'reset_lock'):
        fsspec.asyn.reset_lock()
    else:
        fsspec.asyn.iothread[0] = None
        fsspec.asyn.loop[0] = None
        fsspec.asyn.lock = threading.Lock()