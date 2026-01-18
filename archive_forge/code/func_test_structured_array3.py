import numpy as np
import ctypes
from numba.cuda.cudadrv.devicearray import (DeviceRecord, from_record_like,
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from numba.np import numpy_support
from numba import cuda
def test_structured_array3(self):
    ary = self.samplerecmat
    mat = np.array([[5.0, 10.0, 15.0], [20.0, 25.0, 30.0], [35.0, 40.0, 45.0]], dtype=np.float32).reshape(3, 3)
    ary['j'][:] = mat
    np.testing.assert_equal(ary['j'], mat)