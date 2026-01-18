import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_record_write_1d_array(self):
    """
        Test writing to a 1D array within a structured type
        """
    rec = self.samplerec1darr.copy()
    nbrecord = numpy_support.from_dtype(recordwitharray)
    cfunc = self.get_cfunc(record_write_array, (nbrecord,))
    cfunc[1, 1](rec)
    expected = self.samplerec1darr.copy()
    expected['g'] = 2
    expected['h'][0] = 3.0
    expected['h'][1] = 4.0
    np.testing.assert_equal(expected, rec)