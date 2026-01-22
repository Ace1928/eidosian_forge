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
@unittest.skipIf(config.NUMBA_NUM_THREADS < 3, 'Not enough CPU cores')
def _test_nested_parallelism_3(self):
    if threading_layer() == 'workqueue':
        self.skipTest('workqueue is not threadsafe')
    BIG = 1000000

    @njit(parallel=True)
    def work(local_nt):
        tid = np.zeros(BIG)
        acc = 0
        set_num_threads(local_nt)
        for i in prange(BIG):
            acc += 1
            tid[i] = get_thread_id()
        return (acc, np.unique(tid))

    @njit(parallel=True)
    def test_func_jit(nthreads):
        set_num_threads(nthreads)
        lens = np.zeros(nthreads)
        total = 0
        for i in prange(nthreads):
            my_acc, tids = work(nthreads + 1)
            lens[i] = len(tids)
            total += my_acc
        return (total, np.unique(lens))
    NT = 2
    expected_acc = BIG * NT
    expected_thread_count = NT + 1
    got_acc, got_tc = test_func_jit(NT)
    self.assertEqual(expected_acc, got_acc)
    self.check_mask(expected_thread_count, got_tc)

    def test_guvectorize(nthreads):

        @guvectorize(['int64[:], int64[:]'], '(n), (n)', nopython=True, target='parallel')
        def test_func_guvectorize(total, lens):
            my_acc, tids = work(nthreads + 1)
            lens[0] = len(tids)
            total[0] += my_acc
        total = np.zeros((nthreads, 1), dtype=np.int64)
        lens = np.zeros(nthreads, dtype=np.int64).reshape((nthreads, 1))
        test_func_guvectorize(total, lens)
        return (total.sum(), np.unique(lens))
    got_acc, got_tc = test_guvectorize(NT)
    self.assertEqual(expected_acc, got_acc)
    self.check_mask(expected_thread_count, got_tc)