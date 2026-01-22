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
@unittest.skipIf(not sys.platform.startswith('linux'), 'Linux only')
def _test_threadmask_across_fork(self):
    forkctx = multiprocessing.get_context('fork')

    @njit
    def foo():
        return get_num_threads()

    def wrap(queue):
        queue.put(foo())
    mask = 1
    self.assertEqual(foo(), config.NUMBA_NUM_THREADS)
    set_num_threads(mask)
    self.assertEqual(foo(), mask)
    shared_queue = forkctx.Queue()
    p = forkctx.Process(target=wrap, args=(shared_queue,))
    p.start()
    p.join()
    self.assertEqual(shared_queue.get(), mask)