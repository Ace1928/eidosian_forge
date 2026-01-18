import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
@unittest.expectedFailure
def test_set_arrays(self):
    arr = np.zeros(2, dtype=recordwith2darray).view(np.recarray)
    nbarr = np.zeros(2, dtype=recordwith2darray).view(np.recarray)
    for pyfunc in (recarray_write_array_of_nestedarray_broadcast, recarray_write_array_of_nestedarray):
        arr_expected = pyfunc(arr)
        cfunc = self.get_cfunc(pyfunc, nbarr.dtype)
        arr_res = cfunc(nbarr)
        np.testing.assert_equal(arr_res, arr_expected)