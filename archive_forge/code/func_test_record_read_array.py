import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_record_read_array(self):
    nbval = np.recarray(1, dtype=recordwitharray)
    nbval[0].h[0] = 15.0
    nbval[0].h[1] = 25.0
    cfunc = self.get_cfunc(record_read_array0, np.float32)
    res = cfunc(nbval[0])
    np.testing.assert_equal(res, nbval[0].h[0])
    cfunc = self.get_cfunc(record_read_array1, np.float32)
    res = cfunc(nbval[0])
    np.testing.assert_equal(res, nbval[0].h[1])