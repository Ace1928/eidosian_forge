import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def unmap(self, store, key, check_remap=True):
    """Remove key from this node and its children."""
    if not len(self._items):
        raise AssertionError("can't unmap in an empty InternalNode.")
    children = [node for node, _ in self._iter_nodes(store, key_filter=[key])]
    if children:
        child = children[0]
    else:
        raise KeyError(key)
    self._len -= 1
    unmapped = child.unmap(store, key)
    self._key = None
    search_key = self._search_key(key)
    if len(unmapped) == 0:
        del self._items[search_key]
        unmapped = None
    else:
        self._items[search_key] = unmapped
    if len(self._items) == 1:
        return list(self._items.values())[0]
    if isinstance(unmapped, InternalNode):
        return self
    if check_remap:
        return self._check_remap(store)
    else:
        return self