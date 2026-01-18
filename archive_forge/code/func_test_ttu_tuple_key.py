import math
import unittest
from cachetools import TLRUCache
from . import CacheTestMixin
def test_ttu_tuple_key(self):
    cache = TLRUCache(maxsize=1, ttu=lambda k, v, t: t + 1, timer=Timer())
    cache[1, 2, 3] = 42
    self.assertEqual(42, cache[1, 2, 3])
    cache.timer.tick()
    with self.assertRaises(KeyError):
        cache[1, 2, 3]
    self.assertNotIn((1, 2, 3), cache)