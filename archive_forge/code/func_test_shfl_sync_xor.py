import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def test_shfl_sync_xor(self):
    compiled = cuda.jit('void(int32[:], int32)')(use_shfl_sync_xor)
    nelem = 32
    xor = 16
    ary = np.empty(nelem, dtype=np.int32)
    exp = np.arange(nelem, dtype=np.int32) ^ xor
    compiled[1, nelem](ary, xor)
    self.assertTrue(np.all(ary == exp))