import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import unittest, CUDATestCase
class TestCudaTestGlobal(CUDATestCase):

    def test_global_int_const(self):
        """Test simple_smem
        """
        compiled = cuda.jit('void(int32[:])')(simple_smem)
        nelem = 100
        ary = np.empty(nelem, dtype=np.int32)
        compiled[1, nelem](ary)
        self.assertTrue(np.all(ary == np.arange(nelem, dtype=np.int32)))

    @unittest.SkipTest
    def test_global_tuple_const(self):
        """Test coop_smem2d
        """
        compiled = cuda.jit('void(float32[:,:])')(coop_smem2d)
        shape = (10, 20)
        ary = np.empty(shape, dtype=np.float32)
        compiled[1, shape](ary)
        exp = np.empty_like(ary)
        for i in range(ary.shape[0]):
            for j in range(ary.shape[1]):
                exp[i, j] = float(i + 1) / (j + 1)
        self.assertTrue(np.allclose(ary, exp))