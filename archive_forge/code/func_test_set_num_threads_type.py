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
def test_set_num_threads_type(self):

    @njit
    def foo():
        set_num_threads('wrong_type')
    expected = 'The number of threads specified must be an integer'
    for fn, errty in ((foo, TypingError), (foo.py_func, TypeError)):
        with self.assertRaises(errty) as raises:
            fn()
        self.assertIn(expected, str(raises.exception))