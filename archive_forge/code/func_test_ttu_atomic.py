import math
import unittest
from cachetools import TLRUCache
from . import CacheTestMixin
def test_ttu_atomic(self):
    cache = TLRUCache(maxsize=1, ttu=lambda k, v, t: t + 2, timer=Timer(auto=True))
    cache[1] = 1
    self.assertEqual(1, cache[1])
    cache[1] = 1
    self.assertEqual(1, cache.get(1))
    cache[1] = 1
    self.assertEqual(1, cache.pop(1))
    cache[1] = 1
    self.assertEqual(1, cache.setdefault(1))
    cache[1] = 1
    cache.clear()
    self.assertEqual(0, len(cache))