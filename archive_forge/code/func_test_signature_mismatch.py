import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_signature_mismatch(self):

    @jit(nopython=True)
    def f1(x):
        return x

    @jit(nopython=True)
    def f2(x):
        return x

    @jit(nopython=True)
    def foo(disp1, disp2, sel):
        if sel == 1:
            fn = disp1
        else:
            fn = disp2
        return (fn([1]), fn(2))
    with self.assertRaises(errors.UnsupportedError) as cm:
        foo(f1, f2, sel=1)
    self.assertRegex(str(cm.exception), 'mismatch of function types:')
    self.assertEqual(foo(f1, f1, sel=1), ([1], 2))