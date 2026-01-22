import io
import torch
from ._utils import _type, _cuda, _hpu
from torch.types import Storage
from typing import cast, Any, Dict as _Dict, Optional as _Optional, TypeVar, Type, Union
import copy
import collections
from functools import lru_cache
import warnings
import threading
import functools
def _share_memory_lock_protected(fn):

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        to_free = None
        to_wait = None
        with _share_memory_lock:
            key = self._cdata
            if key in _share_memory_map:
                to_wait = _share_memory_map[key]
            else:
                _share_memory_map[key] = threading.RLock()
                _share_memory_map[key].acquire()
                to_free = key
        if to_wait is not None:
            with to_wait:
                pass
        try:
            return fn(self, *args, **kwargs)
        finally:
            if to_free is not None:
                assert self._cdata == to_free
                with _share_memory_lock:
                    _share_memory_map[to_free].release()
                    del _share_memory_map[to_free]
    return wrapper