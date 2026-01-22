import unittest
def _test_getsizeof(self, cache):
    self.assertEqual(0, cache.currsize)
    self.assertEqual(3, cache.maxsize)
    self.assertEqual(1, cache.getsizeof(1))
    self.assertEqual(2, cache.getsizeof(2))
    self.assertEqual(3, cache.getsizeof(3))
    cache.update({1: 1, 2: 2})
    self.assertEqual(2, len(cache))
    self.assertEqual(3, cache.currsize)
    self.assertEqual(1, cache[1])
    self.assertEqual(2, cache[2])
    cache[1] = 2
    self.assertEqual(1, len(cache))
    self.assertEqual(2, cache.currsize)
    self.assertEqual(2, cache[1])
    self.assertNotIn(2, cache)
    cache.update({1: 1, 2: 2})
    self.assertEqual(2, len(cache))
    self.assertEqual(3, cache.currsize)
    self.assertEqual(1, cache[1])
    self.assertEqual(2, cache[2])
    cache[3] = 3
    self.assertEqual(1, len(cache))
    self.assertEqual(3, cache.currsize)
    self.assertEqual(3, cache[3])
    self.assertNotIn(1, cache)
    self.assertNotIn(2, cache)
    with self.assertRaises(ValueError):
        cache[3] = 4
    self.assertEqual(1, len(cache))
    self.assertEqual(3, cache.currsize)
    self.assertEqual(3, cache[3])
    with self.assertRaises(ValueError):
        cache[4] = 4
    self.assertEqual(1, len(cache))
    self.assertEqual(3, cache.currsize)
    self.assertEqual(3, cache[3])