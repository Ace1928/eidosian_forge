import math
import unittest
from cachetools import TLRUCache
from . import CacheTestMixin
def test_ttu_expired(self):
    cache = TLRUCache(maxsize=1, ttu=lambda k, _, t: t + k, timer=Timer())
    cache[1] = None
    self.assertEqual(cache[1], None)
    self.assertEqual(1, len(cache))
    cache[0] = None
    self.assertNotIn(0, cache)
    self.assertEqual(cache[1], None)
    self.assertEqual(1, len(cache))
    cache[-1] = None
    self.assertNotIn(-1, cache)
    self.assertNotIn(0, cache)
    self.assertEqual(cache[1], None)
    self.assertEqual(1, len(cache))