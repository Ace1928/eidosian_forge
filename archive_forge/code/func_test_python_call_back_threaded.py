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
def test_python_call_back_threaded(self):

    def pyfunc(a, repeat):
        out = 0
        for _ in range(repeat):
            out += py_call_back(a)
        return out
    cfunc = jit(nopython=True, nogil=True)(pyfunc)
    arr = np.array(['what'], dtype='S10')
    repeat = 1000
    expected = pyfunc(arr, repeat)
    outputs = []
    cfunc(arr, repeat)

    def run(func, arr, repeat):
        outputs.append(func(arr, repeat))
    threads = [threading.Thread(target=run, args=(cfunc, arr, repeat)) for _ in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for got in outputs:
        self.assertEqual(expected, got)