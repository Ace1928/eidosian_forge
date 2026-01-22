import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _current_size(self):
    """Answer the current serialised size of this node."""
    return self._raw_size + len(str(self._len)) + len(str(self._key_width)) + len(str(self._maximum_size))