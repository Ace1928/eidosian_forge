import inspect
import llvmlite.binding as ll
import multiprocessing
import numpy as np
import os
import stat
import shutil
import subprocess
import sys
import traceback
import unittest
import warnings
from numba import njit
from numba.core import codegen
from numba.core.caching import _UserWideCacheLocator
from numba.core.errors import NumbaWarning
from numba.parfors import parfor
from numba.tests.support import (
from numba import njit
from numba import njit
from file2 import function2
from numba import njit
class DispatcherCacheUsecasesTest(BaseCacheTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, 'cache_usecases.py')
    modname = 'dispatcher_caching_test_fodder'

    def run_in_separate_process(self, *, envvars={}):
        code = 'if 1:\n            import sys\n\n            sys.path.insert(0, %(tempdir)r)\n            mod = __import__(%(modname)r)\n            mod.self_test()\n            ' % dict(tempdir=self.tempdir, modname=self.modname)
        subp_env = os.environ.copy()
        subp_env.update(envvars)
        popen = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=subp_env)
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError('process failed with code %s: \nstdout follows\n%s\nstderr follows\n%s\n' % (popen.returncode, out.decode(), err.decode()))

    def check_hits(self, func, hits, misses=None):
        st = func.stats
        self.assertEqual(sum(st.cache_hits.values()), hits, st.cache_hits)
        if misses is not None:
            self.assertEqual(sum(st.cache_misses.values()), misses, st.cache_misses)