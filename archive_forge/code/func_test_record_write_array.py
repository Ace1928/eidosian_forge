import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_record_write_array(self):
    nbval = np.recarray(1, dtype=recordwitharray)
    nbrecord = numpy_support.from_dtype(recordwitharray)
    cfunc = self.get_cfunc(record_write_array, (nbrecord,))
    cfunc(nbval[0])
    expected = np.recarray(1, dtype=recordwitharray)
    expected[0].g = 2
    expected[0].h[0] = 3.0
    expected[0].h[1] = 4.0
    np.testing.assert_equal(expected, nbval)