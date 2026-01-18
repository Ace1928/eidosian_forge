import math
import unittest
from cachetools import TLRUCache
from . import CacheTestMixin
def test_ttu_expire(self):
    cache = TLRUCache(maxsize=3, ttu=lambda k, v, t: t + 3, timer=Timer())
    with cache.timer as time:
        self.assertEqual(time, cache.timer())
    cache[1] = 1
    cache.timer.tick()
    cache[2] = 2
    cache.timer.tick()
    cache[3] = 3
    self.assertEqual(2, cache.timer())
    self.assertEqual({1, 2, 3}, set(cache))
    self.assertEqual(3, len(cache))
    self.assertEqual(1, cache[1])
    self.assertEqual(2, cache[2])
    self.assertEqual(3, cache[3])
    cache.expire()
    self.assertEqual({1, 2, 3}, set(cache))
    self.assertEqual(3, len(cache))
    self.assertEqual(1, cache[1])
    self.assertEqual(2, cache[2])
    self.assertEqual(3, cache[3])
    cache.expire(3)
    self.assertEqual({2, 3}, set(cache))
    self.assertEqual(2, len(cache))
    self.assertNotIn(1, cache)
    self.assertEqual(2, cache[2])
    self.assertEqual(3, cache[3])
    cache.expire(4)
    self.assertEqual({3}, set(cache))
    self.assertEqual(1, len(cache))
    self.assertNotIn(1, cache)
    self.assertNotIn(2, cache)
    self.assertEqual(3, cache[3])
    cache.expire(5)
    self.assertEqual(set(), set(cache))
    self.assertEqual(0, len(cache))
    self.assertNotIn(1, cache)
    self.assertNotIn(2, cache)
    self.assertNotIn(3, cache)