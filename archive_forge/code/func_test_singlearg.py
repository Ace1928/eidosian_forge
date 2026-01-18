import random
import time
import unittest
def test_singlearg(self):
    cache = DummyLRUCache()
    decorator = self._makeOne(0, cache)

    def wrapped(key):
        return key
    decorated = decorator(wrapped)
    result = decorated(1)
    self.assertEqual(cache[1,], 1)
    self.assertEqual(result, 1)
    self.assertEqual(len(cache), 1)
    result = decorated(2)
    self.assertEqual(cache[2,], 2)
    self.assertEqual(result, 2)
    self.assertEqual(len(cache), 2)
    result = decorated(2)
    self.assertEqual(cache[2,], 2)
    self.assertEqual(result, 2)
    self.assertEqual(len(cache), 2)