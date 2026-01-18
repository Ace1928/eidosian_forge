import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_unless_cudasim
import numba.cuda.simulator as simulator
import unittest
def test_recarray_setting(self):
    recordwith2darray = np.dtype([('i', np.int32), ('j', np.float32, (3, 2))])
    rec = np.recarray(2, dtype=recordwith2darray)
    rec[0]['i'] = 45

    @cuda.jit
    def simple_kernel(f):
        f[1] = f[0]
    simple_kernel[1, 1](rec)
    np.testing.assert_equal(rec[0]['i'], rec[1]['i'])