import math
import unittest
from cachetools import TLRUCache
from . import CacheTestMixin
def test_ttu_lru(self):
    cache = TLRUCache(maxsize=2, ttu=lambda k, v, t: t + 1, timer=Timer())
    self.assertEqual(0, cache.timer())
    self.assertEqual(2, cache.ttu(None, None, 1))
    cache[1] = 1
    cache[2] = 2
    cache[3] = 3
    self.assertEqual(len(cache), 2)
    self.assertNotIn(1, cache)
    self.assertEqual(cache[2], 2)
    self.assertEqual(cache[3], 3)
    cache[2]
    cache[4] = 4
    self.assertEqual(len(cache), 2)
    self.assertNotIn(1, cache)
    self.assertEqual(cache[2], 2)
    self.assertNotIn(3, cache)
    self.assertEqual(cache[4], 4)
    cache[5] = 5
    self.assertEqual(len(cache), 2)
    self.assertNotIn(1, cache)
    self.assertNotIn(2, cache)
    self.assertNotIn(3, cache)
    self.assertEqual(cache[4], 4)
    self.assertEqual(cache[5], 5)