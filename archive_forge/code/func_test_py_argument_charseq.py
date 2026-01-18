import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_py_argument_charseq(self):
    pyfunc = set_charseq
    rectype = numpy_support.from_dtype(recordwithcharseq)
    sig = (rectype[::1], types.intp, rectype.typeof('n'))
    cfunc = njit(sig)(pyfunc).overloads[sig].entry_point
    for i in range(self.refsample1d.size):
        chars = '{0}'.format(hex(i + 10))
        pyfunc(self.refsample1d, i, chars)
        cfunc(self.nbsample1d, i, chars)
        np.testing.assert_equal(self.refsample1d, self.nbsample1d)