import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _search_key(self, key):
    """Return the serialised key for key in this node."""
    return (self._search_key_func(key) + b'\x00' * self._node_width)[:self._node_width]