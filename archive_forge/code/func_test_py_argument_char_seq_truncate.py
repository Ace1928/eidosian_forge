import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_py_argument_char_seq_truncate(self):
    pyfunc = set_charseq
    rectype = numpy_support.from_dtype(recordwithcharseq)
    sig = (rectype[::1], types.intp, rectype.typeof('n'))
    cfunc = njit(sig)(pyfunc).overloads[sig].entry_point
    cs_overflowed = 'abcdef'
    pyfunc(self.refsample1d, 1, cs_overflowed)
    cfunc(self.nbsample1d, 1, cs_overflowed)
    np.testing.assert_equal(self.refsample1d, self.nbsample1d)
    self.assertEqual(self.refsample1d[1].n, cs_overflowed[:-1].encode('ascii'))