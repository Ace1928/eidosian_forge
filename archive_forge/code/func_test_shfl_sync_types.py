import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def test_shfl_sync_types(self):
    types = (int32, int64, float32, float64)
    values = (np.int32(-1), np.int64(1 << 42), np.float32(np.pi), np.float64(np.pi))
    for typ, val in zip(types, values):
        compiled = cuda.jit((typ[:], typ))(use_shfl_sync_with_val)
        nelem = 32
        ary = np.empty(nelem, dtype=val.dtype)
        compiled[1, nelem](ary, val)
        self.assertTrue(np.all(ary == val))