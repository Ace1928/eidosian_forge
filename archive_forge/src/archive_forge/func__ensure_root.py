import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _ensure_root(self):
    """Ensure that the root node is an object not a key."""
    if isinstance(self._root_node, StaticTuple):
        self._root_node = self._get_node(self._root_node)