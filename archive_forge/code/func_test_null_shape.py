import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
from numba import config, cuda
def test_null_shape(self):
    null_shape = ()
    shape1 = cuda.device_array(()).shape
    shape2 = cuda.device_array_like(np.ndarray(())).shape
    self.assertEqual(shape1, null_shape)
    self.assertEqual(shape2, null_shape)