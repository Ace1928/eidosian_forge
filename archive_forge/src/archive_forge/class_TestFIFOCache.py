from .. import fifo_cache, tests
class TestFIFOCache(tests.TestCase):
    """Test that FIFO cache properly keeps track of entries."""

    def test_add_is_present(self):
        c = fifo_cache.FIFOCache()
        c[1] = 2
        self.assertTrue(1 in c)
        self.assertEqual(1, len(c))
        self.assertEqual(2, c[1])
        self.assertEqual(2, c.get(1))
        self.assertEqual(2, c.get(1, None))
        self.assertEqual([1], list(c))
        self.assertEqual({1}, c.keys())
        self.assertEqual([(1, 2)], sorted(c.items()))
        self.assertEqual([2], sorted(c.values()))
        self.assertEqual({1: 2}, c)

    def test_cache_size(self):
        c = fifo_cache.FIFOCache()
        self.assertEqual(100, c.cache_size())
        c.resize(20, 5)
        self.assertEqual(20, c.cache_size())

    def test_missing(self):
        c = fifo_cache.FIFOCache()
        self.assertRaises(KeyError, c.__getitem__, 1)
        self.assertFalse(1 in c)
        self.assertEqual(0, len(c))
        self.assertEqual(None, c.get(1))
        self.assertEqual(None, c.get(1, None))
        self.assertEqual([], list(c))
        self.assertEqual(set(), c.keys())
        self.assertEqual([], list(c.items()))
        self.assertEqual([], list(c.values()))
        self.assertEqual({}, c)

    def test_add_maintains_fifo(self):
        c = fifo_cache.FIFOCache(4, 4)
        c[1] = 2
        c[2] = 3
        c[3] = 4
        c[4] = 5
        self.assertEqual({1, 2, 3, 4}, c.keys())
        c[5] = 6
        self.assertEqual({2, 3, 4, 5}, c.keys())
        c[2] = 7
        self.assertEqual({2, 3, 4, 5}, c.keys())
        c[6] = 7
        self.assertEqual({2, 4, 5, 6}, c.keys())
        self.assertEqual([4, 5, 2, 6], list(c._queue))

    def test_default_after_cleanup_count(self):
        c = fifo_cache.FIFOCache(5)
        self.assertEqual(4, c._after_cleanup_count)
        c[1] = 2
        c[2] = 3
        c[3] = 4
        c[4] = 5
        c[5] = 6
        self.assertEqual({1, 2, 3, 4, 5}, c.keys())
        c[6] = 7
        self.assertEqual({3, 4, 5, 6}, c.keys())

    def test_clear(self):
        c = fifo_cache.FIFOCache(5)
        c[1] = 2
        c[2] = 3
        c[3] = 4
        c[4] = 5
        c[5] = 6
        c.cleanup()
        self.assertEqual({2, 3, 4, 5}, c.keys())
        c.clear()
        self.assertEqual(set(), c.keys())
        self.assertEqual([], list(c._queue))
        self.assertEqual({}, c)

    def test_copy_not_implemented(self):
        c = fifo_cache.FIFOCache()
        self.assertRaises(NotImplementedError, c.copy)

    def test_pop_not_implemeted(self):
        c = fifo_cache.FIFOCache()
        self.assertRaises(NotImplementedError, c.pop, 'key')

    def test_popitem_not_implemeted(self):
        c = fifo_cache.FIFOCache()
        self.assertRaises(NotImplementedError, c.popitem)

    def test_resize_smaller(self):
        c = fifo_cache.FIFOCache()
        c[1] = 2
        c[2] = 3
        c[3] = 4
        c[4] = 5
        c[5] = 6
        c.resize(5)
        self.assertEqual({1: 2, 2: 3, 3: 4, 4: 5, 5: 6}, c)
        self.assertEqual(5, c.cache_size())
        c[6] = 7
        self.assertEqual({3: 4, 4: 5, 5: 6, 6: 7}, c)
        c.resize(3, 2)
        self.assertEqual({5: 6, 6: 7}, c)

    def test_resize_larger(self):
        c = fifo_cache.FIFOCache(5, 4)
        c[1] = 2
        c[2] = 3
        c[3] = 4
        c[4] = 5
        c[5] = 6
        c.resize(10)
        self.assertEqual({1: 2, 2: 3, 3: 4, 4: 5, 5: 6}, c)
        self.assertEqual(10, c.cache_size())
        c[6] = 7
        c[7] = 8
        c[8] = 9
        c[9] = 10
        c[10] = 11
        self.assertEqual({1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11}, c)
        c[11] = 12
        self.assertEqual({4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12}, c)

    def test_setdefault(self):
        c = fifo_cache.FIFOCache(5, 4)
        c['one'] = 1
        c['two'] = 2
        c['three'] = 3
        myobj = object()
        self.assertIs(myobj, c.setdefault('four', myobj))
        self.assertEqual({'one': 1, 'two': 2, 'three': 3, 'four': myobj}, c)
        self.assertEqual(3, c.setdefault('three', myobj))
        c.setdefault('five', myobj)
        c.setdefault('six', myobj)
        self.assertEqual({'three': 3, 'four': myobj, 'five': myobj, 'six': myobj}, c)

    def test_update(self):
        c = fifo_cache.FIFOCache(5, 4)
        c.update([(1, 2), (3, 4)])
        self.assertEqual({1: 2, 3: 4}, c)
        c.update(foo=3, bar=4)
        self.assertEqual({1: 2, 3: 4, 'foo': 3, 'bar': 4}, c)
        c.update({'baz': 'biz', 'bing': 'bang'})
        self.assertEqual({'foo': 3, 'bar': 4, 'baz': 'biz', 'bing': 'bang'}, c)
        self.assertRaises(TypeError, c.update, [(1, 2)], [(3, 4)])
        c.update([('a', 'b'), ('d', 'e')], a='c', q='r')
        self.assertEqual({'baz': 'biz', 'bing': 'bang', 'a': 'c', 'd': 'e', 'q': 'r'}, c)

    def test_cleanup_funcs(self):
        log = []

        def logging_cleanup(key, value):
            log.append((key, value))
        c = fifo_cache.FIFOCache(5, 4)
        c.add(1, 2, cleanup=logging_cleanup)
        c.add(2, 3, cleanup=logging_cleanup)
        c.add(3, 4, cleanup=logging_cleanup)
        c.add(4, 5, cleanup=None)
        c[5] = 6
        self.assertEqual([], log)
        c.add(6, 7, cleanup=logging_cleanup)
        self.assertEqual([(1, 2), (2, 3)], log)
        del log[:]
        c.add(3, 8, cleanup=logging_cleanup)
        self.assertEqual([(3, 4)], log)
        del log[:]
        c[3] = 9
        self.assertEqual([(3, 8)], log)
        del log[:]
        c.clear()
        self.assertEqual([(6, 7)], log)
        del log[:]
        c.add(8, 9, cleanup=logging_cleanup)
        del c[8]
        self.assertEqual([(8, 9)], log)

    def test_cleanup_at_deconstruct(self):
        log = []

        def logging_cleanup(key, value):
            log.append((key, value))
        c = fifo_cache.FIFOCache()
        c.add(1, 2, cleanup=logging_cleanup)
        del c
        self.assertEqual([], log)