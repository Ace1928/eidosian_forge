import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def test_shfl_sync_down(self):
    compiled = cuda.jit('void(int32[:], int32)')(use_shfl_sync_down)
    nelem = 32
    delta = 4
    ary = np.empty(nelem, dtype=np.int32)
    exp = np.arange(nelem, dtype=np.int32)
    exp[:-delta] += delta
    compiled[1, nelem](ary, delta)
    self.assertTrue(np.all(ary == exp))