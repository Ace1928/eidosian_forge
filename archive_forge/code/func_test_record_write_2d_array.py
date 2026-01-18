import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_record_write_2d_array(self):
    """
        Test writing to a 2D array within a structured type
        """
    rec = self.samplerec2darr.copy()
    nbrecord = numpy_support.from_dtype(recordwith2darray)
    cfunc = self.get_cfunc(record_write_2d_array, (nbrecord,))
    cfunc[1, 1](rec)
    expected = self.samplerec2darr.copy()
    expected['i'] = 3
    expected['j'][:] = np.asarray([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], np.float32).reshape(3, 2)
    np.testing.assert_equal(expected, rec)