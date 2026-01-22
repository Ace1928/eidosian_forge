import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def _test_rec_set(self, v, pyfunc, f):
    rec = self.sample1d.copy()[0]
    nbrecord = numpy_support.from_dtype(recordtype)
    cfunc = self.get_cfunc(pyfunc, (nbrecord,))
    cfunc[1, 1](rec, v)
    np.testing.assert_equal(rec[f], v)