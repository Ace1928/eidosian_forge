import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def test_vote_sync_ballot(self):
    compiled = cuda.jit('void(uint32[:])')(use_vote_sync_ballot)
    nelem = 32
    ary = np.empty(nelem, dtype=np.uint32)
    compiled[1, nelem](ary)
    self.assertTrue(np.all(ary == np.uint32(4294967295)))