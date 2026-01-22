import sys
import os.path
import re
import subprocess
import numpy as np
from numba.tests.support import capture_cache_log
from numba.tests.test_caching import BaseCacheTest
from numba.core import config
import unittest
class UfuncCacheTest(BaseCacheTest):
    """
    Since the cache stats is not exposed by ufunc, we test by looking at the
    cache debug log.
    """
    _numba_parallel_test_ = False
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, 'cache_usecases.py')
    modname = 'ufunc_caching_test_fodder'
    regex_data_saved = re.compile('\\[cache\\] data saved to')
    regex_index_saved = re.compile('\\[cache\\] index saved to')
    regex_data_loaded = re.compile('\\[cache\\] data loaded from')
    regex_index_loaded = re.compile('\\[cache\\] index loaded from')

    def check_cache_saved(self, cachelog, count):
        """
        Check number of cache-save were issued
        """
        data_saved = self.regex_data_saved.findall(cachelog)
        index_saved = self.regex_index_saved.findall(cachelog)
        self.assertEqual(len(data_saved), count)
        self.assertEqual(len(index_saved), count)

    def check_cache_loaded(self, cachelog, count):
        """
        Check number of cache-load were issued
        """
        data_loaded = self.regex_data_loaded.findall(cachelog)
        index_loaded = self.regex_index_loaded.findall(cachelog)
        self.assertEqual(len(data_loaded), count)
        self.assertEqual(len(index_loaded), count)

    def check_ufunc_cache(self, usecase_name, n_overloads, **kwargs):
        """
        Check number of cache load/save.
        There should be one per overloaded version.
        """
        mod = self.import_module()
        usecase = getattr(mod, usecase_name)
        with capture_cache_log() as out:
            new_ufunc = usecase(**kwargs)
        cachelog = out.getvalue()
        self.check_cache_saved(cachelog, count=n_overloads)
        with capture_cache_log() as out:
            cached_ufunc = usecase(**kwargs)
        cachelog = out.getvalue()
        self.check_cache_loaded(cachelog, count=n_overloads)
        return (new_ufunc, cached_ufunc)