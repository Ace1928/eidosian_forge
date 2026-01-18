import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
def test_ufunc_with_allocated_output(self):

    def pyfunc(a, b):
        out = np.empty(a.shape)
        np.add(a, b, out)
        return out
    cfunc = nrtjit(pyfunc)
    arr_a = np.random.random(10)
    arr_b = np.random.random(10)
    np.testing.assert_equal(pyfunc(arr_a, arr_b), cfunc(arr_a, arr_b))
    self.assert_array_nrt_refct(cfunc(arr_a, arr_b), 1)
    arr_a = np.random.random(10).reshape(2, 5)
    arr_b = np.random.random(10).reshape(2, 5)
    np.testing.assert_equal(pyfunc(arr_a, arr_b), cfunc(arr_a, arr_b))
    self.assert_array_nrt_refct(cfunc(arr_a, arr_b), 1)
    arr_a = np.random.random(70).reshape(2, 5, 7)
    arr_b = np.random.random(70).reshape(2, 5, 7)
    np.testing.assert_equal(pyfunc(arr_a, arr_b), cfunc(arr_a, arr_b))
    self.assert_array_nrt_refct(cfunc(arr_a, arr_b), 1)