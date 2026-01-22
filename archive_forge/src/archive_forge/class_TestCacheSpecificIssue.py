import sys
import os.path
import re
import subprocess
import numpy as np
from numba.tests.support import capture_cache_log
from numba.tests.test_caching import BaseCacheTest
from numba.core import config
import unittest
class TestCacheSpecificIssue(UfuncCacheTest):

    def run_in_separate_process(self, runcode):
        code = 'if 1:\n            import sys\n\n            sys.path.insert(0, %(tempdir)r)\n            mod = __import__(%(modname)r)\n            mod.%(runcode)s\n            ' % dict(tempdir=self.tempdir, modname=self.modname, runcode=runcode)
        popen = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError('process failed with code %s: stderr follows\n%s\n' % (popen.returncode, err.decode()))

    def test_first_load_cached_ufunc(self):
        self.run_in_separate_process('direct_ufunc_cache_usecase()')
        self.run_in_separate_process('direct_ufunc_cache_usecase()')

    def test_first_load_cached_gufunc(self):
        self.run_in_separate_process('direct_gufunc_cache_usecase()')
        self.run_in_separate_process('direct_gufunc_cache_usecase()')