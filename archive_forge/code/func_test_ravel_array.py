from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_ravel_array(self, flags=enable_pyobj_flags):

    def generic_check(pyfunc, a, assume_layout):
        arraytype1 = typeof(a)
        self.assertEqual(arraytype1.layout, assume_layout)
        cfunc = jit((arraytype1,), **flags)(pyfunc)
        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)
        py_copied = a.ctypes.data != expected.ctypes.data
        nb_copied = a.ctypes.data != got.ctypes.data
        self.assertEqual(py_copied, assume_layout != 'C')
        self.assertEqual(py_copied, nb_copied)
    check_method = partial(generic_check, ravel_array)
    check_function = partial(generic_check, numpy_ravel_array)

    def check(*args, **kwargs):
        check_method(*args, **kwargs)
        check_function(*args, **kwargs)
    check(np.arange(9).reshape(3, 3), assume_layout='C')
    check(np.arange(9).reshape(3, 3, order='F'), assume_layout='F')
    check(np.arange(18).reshape(3, 3, 2)[:, :, 0], assume_layout='A')
    check(np.arange(18).reshape(2, 3, 3), assume_layout='C')
    check(np.arange(18).reshape(2, 3, 3, order='F'), assume_layout='F')
    check(np.arange(36).reshape(2, 3, 3, 2)[:, :, :, 0], assume_layout='A')