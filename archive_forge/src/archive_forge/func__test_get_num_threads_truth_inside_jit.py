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
def _test_get_num_threads_truth_inside_jit(self):
    for mask in range(2, min(6, config.NUMBA_NUM_THREADS + 1)):

        @njit(parallel=True)
        def test_func():
            set_num_threads(mask)
            x = 5000000
            buf = np.empty((x,))
            for i in prange(x):
                buf[i] = get_thread_id()
            return (len(np.unique(buf)), get_num_threads())
        out = test_func()
        self.check_mask((mask, mask), out)

        @guvectorize(['void(int64[:], int64[:])'], '(n), (m)', nopython=True, target='parallel')
        def test_gufunc(x, out):
            set_num_threads(mask)
            x[:] = get_thread_id()
            out[0] = get_num_threads()
        x = np.full((5000000,), -1, dtype=np.int64).reshape((100, 50000))
        out = np.zeros((1,), dtype=np.int64)
        test_gufunc(x, out)
        self.check_mask(mask, out)
        self.check_mask(mask, len(np.unique(x)))