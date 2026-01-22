import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _process_queues(self):
    while self._old_queue:
        self._process_next_old()
    return self._flush_new_queue()