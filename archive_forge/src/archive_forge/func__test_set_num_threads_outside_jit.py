from __future__ import print_function, absolute_import, division
import sys
import os
import re
import multiprocessing
import unittest
import numpy as np
from numba import (njit, set_num_threads, get_num_threads, prange, config,
from numba.np.ufunc.parallel import get_thread_id
from numba.core.errors import TypingError
from numba.tests.support import TestCase, skip_parfors_unsupported, tag
from numba.tests.test_parallel_backend import TestInSubprocess
@skip_parfors_unsupported
@unittest.skipIf(config.NUMBA_NUM_THREADS < 2, 'Not enough CPU cores')
def _test_set_num_threads_outside_jit(self):
    set_num_threads(2)

    @njit(parallel=True)
    def test_func():
        x = 5
        buf = np.empty((x,))
        for i in prange(x):
            buf[i] = get_num_threads()
        return buf

    @guvectorize(['void(int64[:])'], '(n)', nopython=True, target='parallel')
    def test_gufunc(x):
        x[:] = get_num_threads()
    out = test_func()
    np.testing.assert_equal(out, 2)
    x = np.zeros((5000000,), dtype=np.int64)
    test_gufunc(x)
    np.testing.assert_equal(x, 2)