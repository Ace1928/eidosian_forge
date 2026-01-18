import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_record_two_arrays(self):
    nbrecord = numpy_support.from_dtype(recordwith2arrays)
    rec = np.recarray(1, dtype=recordwith2arrays)[0]
    rec.k[:] = np.arange(200).reshape(10, 20)
    rec.l[:] = np.arange(72).reshape(6, 12)
    pyfunc = record_read_first_arr
    cfunc = self.get_cfunc(pyfunc, (nbrecord,))
    self.assertEqual(cfunc(rec), pyfunc(rec))
    pyfunc = record_read_second_arr
    cfunc = self.get_cfunc(pyfunc, (nbrecord,))
    self.assertEqual(cfunc(rec), pyfunc(rec))
    pyfunc = set_field_slice
    cfunc = self.get_cfunc(pyfunc, (nbrecord,))
    np.testing.assert_array_equal(cfunc(rec), pyfunc(rec))