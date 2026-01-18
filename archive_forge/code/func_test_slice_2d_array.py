import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_slice_2d_array(self):
    nbarr = np.recarray(2, dtype=recordwith2darray)
    nbarr[0] = np.array([(1, ((1, 2), (4, 5), (2, 3)))], dtype=recordwith2darray)[0]
    funcs = (rec_getitem_field_slice_2d, recarray_getitem_field_slice_2d)
    for arg, pyfunc in zip([nbarr[0], nbarr], funcs):
        ty = typeof(arg)
        arr_expected = pyfunc(arg)
        cfunc = self.get_cfunc(pyfunc, (ty,))
        arr_res = cfunc(arg)
        np.testing.assert_equal(arr_res, arr_expected)