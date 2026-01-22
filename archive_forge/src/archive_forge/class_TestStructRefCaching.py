import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
class TestStructRefCaching(MemoryLeakMixin, TestCase):

    def setUp(self):
        self._cache_dir = temp_directory(TestStructRefCaching.__name__)
        self._cache_override = override_config('CACHE_DIR', self._cache_dir)
        self._cache_override.__enter__()
        warnings.simplefilter('error')
        warnings.filterwarnings(action='ignore', module='typeguard')

    def tearDown(self):
        self._cache_override.__exit__(None, None, None)
        warnings.resetwarnings()

    def test_structref_caching(self):

        def assert_cached(stats):
            self.assertEqual(len(stats.cache_hits), 1)
            self.assertEqual(len(stats.cache_misses), 0)

        def assert_not_cached(stats):
            self.assertEqual(len(stats.cache_hits), 0)
            self.assertEqual(len(stats.cache_misses), 1)

        def check(cached):
            check_make = njit(cache=True)(caching_test_make)
            check_use = njit(cache=True)(caching_test_use)
            vs = np.random.random(3)
            ctr = 17
            factor = 3
            st = check_make(vs, ctr)
            got = check_use(st, factor)
            expect = vs * factor + ctr
            self.assertPreciseEqual(got, expect)
            if cached:
                assert_cached(check_make.stats)
                assert_cached(check_use.stats)
            else:
                assert_not_cached(check_make.stats)
                assert_not_cached(check_use.stats)
        check(cached=False)
        check(cached=True)