from dulwich import lru_cache
from dulwich.tests import TestCase
def test_resize_smaller(self):
    cache = lru_cache.LRUSizeCache(max_size=10, after_cleanup_size=9)
    cache[1] = 'abc'
    cache[2] = 'def'
    cache[3] = 'ghi'
    cache[4] = 'jkl'
    self.assertEqual([2, 3, 4], sorted(cache.keys()))
    cache.resize(max_size=6, after_cleanup_size=4)
    self.assertEqual([4], sorted(cache.keys()))
    cache[5] = 'mno'
    self.assertEqual([4, 5], sorted(cache.keys()))
    cache[6] = 'pqr'
    self.assertEqual([6], sorted(cache.keys()))