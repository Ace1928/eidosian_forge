import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _compute_serialised_prefix(self):
    """Determine the common prefix for serialised keys in this node.

        :return: A bytestring of the longest serialised key prefix that is
            unique within this node.
        """
    serialised_keys = [self._serialise_key(key) for key in self._items]
    self._common_serialised_prefix = self.common_prefix_for_keys(serialised_keys)
    return self._common_serialised_prefix