import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_setitem_static_int_index(self):

    def check(pyfunc):
        self._test_set_equal(pyfunc, 3.1415, types.float64)
        self._test_set_equal(pyfunc, 3.0, types.float32)
    check(setitem_0)
    check(setitem_1)
    check(setitem_2)
    rec = numpy_support.from_dtype(recordtype)
    with self.assertRaises(TypingError) as raises:
        self.get_cfunc(setitem_10, (rec[:], types.intp, types.float64))
    msg = 'Requested index 10 is out of range'
    self.assertIn(msg, str(raises.exception))