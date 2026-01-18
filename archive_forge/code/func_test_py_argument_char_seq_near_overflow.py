import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_py_argument_char_seq_near_overflow(self):
    pyfunc = set_charseq
    rectype = numpy_support.from_dtype(recordwithcharseq)
    sig = (rectype[::1], types.intp, rectype.typeof('n'))
    cfunc = njit(sig)(pyfunc).overloads[sig].entry_point
    cs_near_overflow = 'abcde'
    self.assertEqual(len(cs_near_overflow), recordwithcharseq['n'].itemsize)
    cfunc(self.nbsample1d, 0, cs_near_overflow)
    self.assertEqual(self.nbsample1d[0]['n'].decode('ascii'), cs_near_overflow)
    np.testing.assert_equal(self.refsample1d[1:], self.nbsample1d[1:])