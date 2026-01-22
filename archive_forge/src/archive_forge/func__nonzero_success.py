from __future__ import annotations
import errno
import os
import site
import stat
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional
import platformdirs
from .utils import deprecation
def _nonzero_success(result: int, func: Any, args: Any) -> Any:
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
    return args