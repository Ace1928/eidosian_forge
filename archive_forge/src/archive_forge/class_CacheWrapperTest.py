import unittest
import cachetools
import cachetools.keys
class CacheWrapperTest(unittest.TestCase, DecoratorTestMixin):

    def cache(self, minsize):
        return cachetools.Cache(maxsize=minsize)

    def test_zero_size_cache_decorator(self):
        cache = self.cache(0)
        wrapper = cachetools.cached(cache)(self.func)
        self.assertEqual(len(cache), 0)
        self.assertEqual(wrapper.__wrapped__, self.func)
        self.assertEqual(wrapper(0), 0)
        self.assertEqual(len(cache), 0)

    def test_zero_size_cache_decorator_lock(self):

        class Lock:
            count = 0

            def __enter__(self):
                Lock.count += 1

            def __exit__(self, *exc):
                pass
        cache = self.cache(0)
        wrapper = cachetools.cached(cache, lock=Lock())(self.func)
        self.assertEqual(len(cache), 0)
        self.assertEqual(wrapper.__wrapped__, self.func)
        self.assertEqual(wrapper(0), 0)
        self.assertEqual(len(cache), 0)
        self.assertEqual(Lock.count, 2)