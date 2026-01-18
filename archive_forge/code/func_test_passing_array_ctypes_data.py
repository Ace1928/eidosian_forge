from ctypes import *
import sys
import threading
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.core.typing import ctypes_utils
from numba.tests.support import MemoryLeakMixin, tag, TestCase
from numba.tests.ctypes_usecases import *
import unittest
def test_passing_array_ctypes_data(self):
    """
        Test the ".ctypes.data" attribute of an array can be passed
        as a "void *" parameter.
        """

    def pyfunc(arr):
        return c_take_array_ptr(arr.ctypes.data)
    cfunc = jit(nopython=True, nogil=True)(pyfunc)
    arr = np.arange(5)
    expected = pyfunc(arr)
    got = cfunc(arr)
    self.assertEqual(expected, got)