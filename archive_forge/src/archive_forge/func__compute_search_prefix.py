import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _compute_search_prefix(self, extra_key=None):
    """Return the unique key prefix for this node.

        :return: A bytestring of the longest search key prefix that is
            unique within this node.
        """
    self._search_prefix = self.common_prefix_for_keys(self._items)
    return self._search_prefix