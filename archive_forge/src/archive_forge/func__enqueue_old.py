import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _enqueue_old(self, new_prefixes, old_chks_to_enqueue):
    for prefix, ref in old_chks_to_enqueue:
        not_interesting = True
        for i in range(len(prefix), 0, -1):
            if prefix[:i] in new_prefixes:
                not_interesting = False
                break
        if not_interesting:
            continue
        self._old_queue.append(ref)