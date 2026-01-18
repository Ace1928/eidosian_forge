import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def test_reshape_c(self):
    ary = np.arange(10)
    expect = ary.reshape(2, 5)
    dary = cuda.to_device(ary)
    dary_reshaped = dary.reshape(2, 5)
    got = dary_reshaped.copy_to_host()
    self.assertPreciseEqual(expect, got)