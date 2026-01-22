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
def _test_set_num_threads_basic_guvectorize(self):
    max_threads = config.NUMBA_NUM_THREADS

    @guvectorize(['void(int64[:])'], '(n)', nopython=True, target='parallel')
    def get_n(x):
        x[:] = get_num_threads()
    x = np.zeros((5000000,), dtype=np.int64)
    get_n(x)
    np.testing.assert_equal(x, max_threads)
    set_num_threads(2)
    x = np.zeros((5000000,), dtype=np.int64)
    get_n(x)
    np.testing.assert_equal(x, 2)
    set_num_threads(max_threads)
    x = np.zeros((5000000,), dtype=np.int64)
    get_n(x)
    np.testing.assert_equal(x, max_threads)

    @guvectorize(['void(int64[:])'], '(n)', nopython=True, target='parallel')
    def set_get_n(n):
        set_num_threads(n[0])
        n[:] = get_num_threads()
    x = np.zeros((5000000,), dtype=np.int64)
    x[0] = 2
    set_get_n(x)
    np.testing.assert_equal(x, 2)
    x = np.zeros((5000000,), dtype=np.int64)
    x[0] = max_threads
    set_get_n(x)
    np.testing.assert_equal(x, max_threads)