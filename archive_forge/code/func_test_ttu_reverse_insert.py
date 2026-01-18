import math
import unittest
from cachetools import TLRUCache
from . import CacheTestMixin
def test_ttu_reverse_insert(self):
    cache = TLRUCache(maxsize=4, ttu=lambda k, v, t: t + v, timer=Timer())
    self.assertEqual(0, cache.timer())
    cache[3] = 3
    cache[2] = 2
    cache[1] = 1
    cache[0] = 0
    self.assertEqual({1, 2, 3}, set(cache))
    self.assertEqual(3, len(cache))
    self.assertNotIn(0, cache)
    self.assertEqual(1, cache[1])
    self.assertEqual(2, cache[2])
    self.assertEqual(3, cache[3])
    cache.timer.tick()
    self.assertEqual({2, 3}, set(cache))
    self.assertEqual(2, len(cache))
    self.assertNotIn(0, cache)
    self.assertNotIn(1, cache)
    self.assertEqual(2, cache[2])
    self.assertEqual(3, cache[3])
    cache.timer.tick()
    self.assertEqual({3}, set(cache))
    self.assertEqual(1, len(cache))
    self.assertNotIn(0, cache)
    self.assertNotIn(1, cache)
    self.assertNotIn(2, cache)
    self.assertEqual(3, cache[3])
    cache.timer.tick()
    self.assertEqual(set(), set(cache))
    self.assertEqual(0, len(cache))
    self.assertNotIn(0, cache)
    self.assertNotIn(1, cache)
    self.assertNotIn(2, cache)
    self.assertNotIn(3, cache)