from dulwich import lru_cache
from dulwich.tests import TestCase
class TestLRUSizeCache(TestCase):

    def test_basic_init(self):
        cache = lru_cache.LRUSizeCache()
        self.assertEqual(2048, cache._max_cache)
        self.assertEqual(int(cache._max_size * 0.8), cache._after_cleanup_size)
        self.assertEqual(0, cache._value_size)

    def test_add__null_key(self):
        cache = lru_cache.LRUSizeCache()
        self.assertRaises(ValueError, cache.add, lru_cache._null_key, 1)

    def test_add_tracks_size(self):
        cache = lru_cache.LRUSizeCache()
        self.assertEqual(0, cache._value_size)
        cache.add('my key', 'my value text')
        self.assertEqual(13, cache._value_size)

    def test_remove_tracks_size(self):
        cache = lru_cache.LRUSizeCache()
        self.assertEqual(0, cache._value_size)
        cache.add('my key', 'my value text')
        self.assertEqual(13, cache._value_size)
        node = cache._cache['my key']
        cache._remove_node(node)
        self.assertEqual(0, cache._value_size)

    def test_no_add_over_size(self):
        """Adding a large value may not be cached at all."""
        cache = lru_cache.LRUSizeCache(max_size=10, after_cleanup_size=5)
        self.assertEqual(0, cache._value_size)
        self.assertEqual({}, cache.items())
        cache.add('test', 'key')
        self.assertEqual(3, cache._value_size)
        self.assertEqual({'test': 'key'}, cache.items())
        cache.add('test2', 'key that is too big')
        self.assertEqual(3, cache._value_size)
        self.assertEqual({'test': 'key'}, cache.items())
        cache.add('test3', 'bigkey')
        self.assertEqual(3, cache._value_size)
        self.assertEqual({'test': 'key'}, cache.items())
        cache.add('test4', 'bikey')
        self.assertEqual(3, cache._value_size)
        self.assertEqual({'test': 'key'}, cache.items())

    def test_no_add_over_size_cleanup(self):
        """If a large value is not cached, we will call cleanup right away."""
        cleanup_calls = []

        def cleanup(key, value):
            cleanup_calls.append((key, value))
        cache = lru_cache.LRUSizeCache(max_size=10, after_cleanup_size=5)
        self.assertEqual(0, cache._value_size)
        self.assertEqual({}, cache.items())
        cache.add('test', 'key that is too big', cleanup=cleanup)
        self.assertEqual(0, cache._value_size)
        self.assertEqual({}, cache.items())
        self.assertEqual([('test', 'key that is too big')], cleanup_calls)

    def test_adding_clears_cache_based_on_size(self):
        """The cache is cleared in LRU order until small enough."""
        cache = lru_cache.LRUSizeCache(max_size=20)
        cache.add('key1', 'value')
        cache.add('key2', 'value2')
        cache.add('key3', 'value23')
        self.assertEqual(5 + 6 + 7, cache._value_size)
        cache['key2']
        cache.add('key4', 'value234')
        self.assertEqual(6 + 8, cache._value_size)
        self.assertEqual({'key2': 'value2', 'key4': 'value234'}, cache.items())

    def test_adding_clears_to_after_cleanup_size(self):
        cache = lru_cache.LRUSizeCache(max_size=20, after_cleanup_size=10)
        cache.add('key1', 'value')
        cache.add('key2', 'value2')
        cache.add('key3', 'value23')
        self.assertEqual(5 + 6 + 7, cache._value_size)
        cache['key2']
        cache.add('key4', 'value234')
        self.assertEqual(8, cache._value_size)
        self.assertEqual({'key4': 'value234'}, cache.items())

    def test_custom_sizes(self):

        def size_of_list(lst):
            return sum((len(x) for x in lst))
        cache = lru_cache.LRUSizeCache(max_size=20, after_cleanup_size=10, compute_size=size_of_list)
        cache.add('key1', ['val', 'ue'])
        cache.add('key2', ['val', 'ue2'])
        cache.add('key3', ['val', 'ue23'])
        self.assertEqual(5 + 6 + 7, cache._value_size)
        cache['key2']
        cache.add('key4', ['value', '234'])
        self.assertEqual(8, cache._value_size)
        self.assertEqual({'key4': ['value', '234']}, cache.items())

    def test_cleanup(self):
        cache = lru_cache.LRUSizeCache(max_size=20, after_cleanup_size=10)
        cache.add('key1', 'value')
        cache.add('key2', 'value2')
        cache.add('key3', 'value23')
        self.assertEqual(5 + 6 + 7, cache._value_size)
        cache.cleanup()
        self.assertEqual(7, cache._value_size)

    def test_keys(self):
        cache = lru_cache.LRUSizeCache(max_size=10)
        cache[1] = 'a'
        cache[2] = 'b'
        cache[3] = 'cdef'
        self.assertEqual([1, 2, 3], sorted(cache.keys()))

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