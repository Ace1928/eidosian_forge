import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _key_value_len(self, key, value):
    return len(self._serialise_key(key)) + 1 + len(b'%d' % value.count(b'\n')) + 1 + len(value) + 1