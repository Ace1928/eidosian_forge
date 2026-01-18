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
def test_python_call_back(self):
    mydct = {'what': 1232121}

    def call_me_maybe(arr):
        return mydct[arr[0].decode('ascii')]
    py_call_back = CFUNCTYPE(c_int, py_object)(call_me_maybe)

    def pyfunc(a):
        what = py_call_back(a)
        return what
    cfunc = jit(nopython=True, nogil=True)(pyfunc)
    arr = np.array(['what'], dtype='S10')
    self.assertEqual(pyfunc(arr), cfunc(arr))