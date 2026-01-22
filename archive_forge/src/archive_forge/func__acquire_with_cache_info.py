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
def _acquire_with_cache_info(self, needs_lock=True):
    """Acquire a file, returning the file and whether it was cached."""
    with self._optional_lock(needs_lock):
        try:
            file = self._cache[self._key]
        except KeyError:
            kwargs = self._kwargs
            if self._mode is not _DEFAULT_MODE:
                kwargs = kwargs.copy()
                kwargs['mode'] = self._mode
            file = self._opener(*self._args, **kwargs)
            if self._mode == 'w':
                self._mode = 'a'
            self._cache[self._key] = file
            return (file, False)
        else:
            return (file, True)