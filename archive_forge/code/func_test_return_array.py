import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
@unittest.expectedFailure
def test_return_array(self):
    nbval = np.recarray(2, dtype=recordwitharray)
    nbval[0] = np.array([(1, (2, 3))], dtype=recordwitharray)[0]
    pyfunc = record_read_array0
    arr_expected = pyfunc(nbval)
    cfunc = self.get_cfunc(pyfunc, np.float32)
    arr_res = cfunc(nbval)
    np.testing.assert_equal(arr_expected, arr_res)