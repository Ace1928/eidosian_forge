import sys
import os.path
import re
import subprocess
import numpy as np
from numba.tests.support import capture_cache_log
from numba.tests.test_caching import BaseCacheTest
from numba.core import config
import unittest
class TestUfuncCacheTest(UfuncCacheTest):

    def test_direct_ufunc_cache(self, **kwargs):
        new_ufunc, cached_ufunc = self.check_ufunc_cache('direct_ufunc_cache_usecase', n_overloads=2, **kwargs)
        inp = np.random.random(10).astype(np.float64)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))
        inp = np.arange(10, dtype=np.intp)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))

    def test_direct_ufunc_cache_objmode(self):
        self.test_direct_ufunc_cache(forceobj=True)

    def test_direct_ufunc_cache_parallel(self):
        self.test_direct_ufunc_cache(target='parallel')

    def test_indirect_ufunc_cache(self, **kwargs):
        new_ufunc, cached_ufunc = self.check_ufunc_cache('indirect_ufunc_cache_usecase', n_overloads=3, **kwargs)
        inp = np.random.random(10).astype(np.float64)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))
        inp = np.arange(10, dtype=np.intp)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))

    def test_indirect_ufunc_cache_parallel(self):
        self.test_indirect_ufunc_cache(target='parallel')