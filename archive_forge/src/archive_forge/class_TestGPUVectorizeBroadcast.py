import numpy as np
from numba import vectorize, cuda
from numba.tests.npyufunc.test_vectorize_decor import BaseVectorizeDecor, \
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestGPUVectorizeBroadcast(CUDATestCase):

    def test_broadcast(self):
        a = np.random.randn(100, 3, 1)
        b = a.transpose(2, 1, 0)

        def fn(a, b):
            return a - b

        @vectorize(['float64(float64,float64)'], target='cuda')
        def fngpu(a, b):
            return a - b
        expect = fn(a, b)
        got = fngpu(a, b)
        np.testing.assert_almost_equal(expect, got)

    def test_device_broadcast(self):
        """
        Same test as .test_broadcast() but with device array as inputs
        """
        a = np.random.randn(100, 3, 1)
        b = a.transpose(2, 1, 0)

        def fn(a, b):
            return a - b

        @vectorize(['float64(float64,float64)'], target='cuda')
        def fngpu(a, b):
            return a - b
        expect = fn(a, b)
        got = fngpu(cuda.to_device(a), cuda.to_device(b))
        np.testing.assert_almost_equal(expect, got.copy_to_host())