import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def _test_rec_read(self, v, pyfunc, f):
    rec = self.sample1d.copy()[0]
    rec[f] = v
    arr = np.zeros(1, v.dtype)
    nbrecord = numpy_support.from_dtype(recordtype)
    cfunc = self.get_cfunc(pyfunc, (nbrecord,))
    cfunc[1, 1](rec, arr)
    np.testing.assert_equal(arr[0], v)