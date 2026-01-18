import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
@unittest.expectedFailure
def test_record_read_arrays(self):
    nbval = np.recarray(2, dtype=recordwitharray)
    nbval[0].h[0] = 15.0
    nbval[0].h[1] = 25.0
    nbval[1].h[0] = 35.0
    nbval[1].h[1] = 45.4
    cfunc = self.get_cfunc(record_read_whole_array, np.float32)
    res = cfunc(nbval)
    np.testing.assert_equal(res, nbval.h)