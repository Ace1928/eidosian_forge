import operator
import unittest
from cachetools import LRUCache, cachedmethod, keys
def test_typed_dict(self):
    cached = Cached(LRUCache(maxsize=2))
    self.assertEqual(cached.get_typed(0), 0)
    self.assertEqual(cached.get_typed(1), 1)
    self.assertEqual(cached.get_typed(1), 1)
    self.assertEqual(cached.get_typed(1.0), 2)
    self.assertEqual(cached.get_typed(1.0), 2)
    self.assertEqual(cached.get_typed(0.0), 3)
    self.assertEqual(cached.get_typed(0), 4)