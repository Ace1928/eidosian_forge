import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def test_useful_syncwarp(self):
    compiled = cuda.jit('void(int32[:])')(useful_syncwarp)
    nelem = 32
    ary = np.empty(nelem, dtype=np.int32)
    compiled[1, nelem](ary)
    self.assertTrue(np.all(ary == 42))