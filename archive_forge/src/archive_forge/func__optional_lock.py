from __future__ import annotations
import atexit
import contextlib
import io
import threading
import uuid
import warnings
from collections.abc import Hashable
from typing import Any
from xarray.backends.locks import acquire
from xarray.backends.lru_cache import LRUCache
from xarray.core import utils
from xarray.core.options import OPTIONS
@contextlib.contextmanager
def _optional_lock(self, needs_lock):
    """Context manager for optionally acquiring a lock."""
    if needs_lock:
        with self._lock:
            yield
    else:
        yield