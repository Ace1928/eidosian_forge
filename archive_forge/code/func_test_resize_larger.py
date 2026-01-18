from dulwich import lru_cache
from dulwich.tests import TestCase
def test_resize_larger(self):
    cache = lru_cache.LRUSizeCache(max_size=10, after_cleanup_size=9)
    cache[1] = 'abc'
    cache[2] = 'def'
    cache[3] = 'ghi'
    cache[4] = 'jkl'
    self.assertEqual([2, 3, 4], sorted(cache.keys()))
    cache.resize(max_size=15, after_cleanup_size=12)
    self.assertEqual([2, 3, 4], sorted(cache.keys()))
    cache[5] = 'mno'
    cache[6] = 'pqr'
    self.assertEqual([2, 3, 4, 5, 6], sorted(cache.keys()))
    cache[7] = 'stu'
    self.assertEqual([4, 5, 6, 7], sorted(cache.keys()))