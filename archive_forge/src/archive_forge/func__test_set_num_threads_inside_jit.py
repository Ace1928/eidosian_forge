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
def _test_set_num_threads_inside_jit(self):

    @njit(parallel=True)
    def test_func(nthreads):
        x = 5
        buf = np.empty((x,))
        set_num_threads(nthreads)
        for i in prange(x):
            buf[i] = get_num_threads()
        return buf
    mask = 2
    out = test_func(mask)
    np.testing.assert_equal(out, mask)