import random
import time
import unittest
class CacherMaker(unittest.TestCase):

    def _getTargetClass(self):
        from repoze.lru import CacheMaker
        return CacheMaker

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_named_cache(self):
        maker = self._makeOne()
        size = 10
        name = 'name'
        decorated = maker.lrucache(maxsize=size, name=name)(_adder)
        self.assertEqual(list(maker._cache.keys()), [name])
        self.assertEqual(maker._cache[name].size, size)
        decorated(10)
        decorated(11)
        self.assertEqual(len(maker._cache[name].data), 2)

    def test_exception(self):
        maker = self._makeOne()
        size = 10
        name = 'name'
        decorated = maker.lrucache(maxsize=size, name=name)(_adder)
        self.assertRaises(KeyError, maker.lrucache, maxsize=size, name=name)
        self.assertRaises(ValueError, maker.lrucache)

    def test_defaultvalue_and_clear(self):
        size = 10
        maker = self._makeOne(maxsize=size)
        for i in range(100):
            decorated = maker.lrucache()(_adder)
            decorated(10)
        self.assertEqual(len(maker._cache), 100)
        for _cache in maker._cache.values():
            self.assertEqual(_cache.size, size)
            self.assertEqual(len(_cache.data), 1)
        maker.clear()
        for _cache in maker._cache.values():
            self.assertEqual(_cache.size, size)
            self.assertEqual(len(_cache.data), 0)

    def test_clear_with_single_name(self):
        maker = self._makeOne(maxsize=10)
        one = maker.lrucache(name='one')(_adder)
        two = maker.lrucache(name='two')(_adder)
        for i in range(100):
            _ = one(i)
            _ = two(i)
        self.assertEqual(len(maker._cache['one'].data), 10)
        self.assertEqual(len(maker._cache['two'].data), 10)
        maker.clear('one')
        self.assertEqual(len(maker._cache['one'].data), 0)
        self.assertEqual(len(maker._cache['two'].data), 10)

    def test_clear_with_multiple_names(self):
        maker = self._makeOne(maxsize=10)
        one = maker.lrucache(name='one')(_adder)
        two = maker.lrucache(name='two')(_adder)
        three = maker.lrucache(name='three')(_adder)
        for i in range(100):
            _ = one(i)
            _ = two(i)
            _ = three(i)
        self.assertEqual(len(maker._cache['one'].data), 10)
        self.assertEqual(len(maker._cache['two'].data), 10)
        self.assertEqual(len(maker._cache['three'].data), 10)
        maker.clear('one', 'three')
        self.assertEqual(len(maker._cache['one'].data), 0)
        self.assertEqual(len(maker._cache['two'].data), 10)
        self.assertEqual(len(maker._cache['three'].data), 0)

    def test_memoized(self):
        from repoze.lru import lru_cache
        from repoze.lru import UnboundedCache
        maker = self._makeOne(maxsize=10)
        memo = maker.memoized('test')
        self.assertIsInstance(memo, lru_cache)
        self.assertIsInstance(memo.cache, UnboundedCache)
        self.assertIs(memo.cache, maker._cache['test'])

    def test_expiring(self):
        size = 10
        timeout = 10
        name = 'name'
        cache = self._makeOne(maxsize=size, timeout=timeout)
        for i in range(100):
            if not i:
                decorator = cache.expiring_lrucache(name=name)
                decorated = decorator(_adder)
                self.assertEqual(cache._cache[name].size, size)
            else:
                decorator = cache.expiring_lrucache()
                decorated = decorator(_adder)
                self.assertEqual(decorator.cache.default_timeout, timeout)
            decorated(10)
        self.assertEqual(len(cache._cache), 100)
        for _cache in cache._cache.values():
            self.assertEqual(_cache.size, size)
            self.assertEqual(_cache.default_timeout, timeout)
            self.assertEqual(len(_cache.data), 1)
        cache.clear()
        for _cache in cache._cache.values():
            self.assertEqual(_cache.size, size)
            self.assertEqual(len(_cache.data), 0)

    def test_expiring_w_timeout(self):
        size = 10
        maker_timeout = 10
        timeout = 20
        name = 'name'
        cache = self._makeOne(maxsize=size, timeout=maker_timeout)
        decorator = cache.expiring_lrucache(name=name, timeout=20)
        self.assertEqual(decorator.cache.default_timeout, timeout)