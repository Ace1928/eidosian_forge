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
def test_storing_voidptr_to_int_array(self):
    cproto = CFUNCTYPE(c_void_p)

    @cproto
    def get_voidstar():
        return 3735928559

    def pyfunc(a):
        ptr = get_voidstar()
        a[0] = ptr
        return ptr
    cfunc = njit((types.uintp[::1],))(pyfunc)
    arr_got = np.zeros(1, dtype=np.uintp)
    arr_expect = arr_got.copy()
    ret_got = cfunc(arr_got)
    ret_expect = pyfunc(arr_expect)
    self.assertEqual(ret_expect, 3735928559)
    self.assertPreciseEqual(ret_got, ret_expect)
    self.assertPreciseEqual(arr_got, arr_expect)