import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
class TestFreeVar(CUDATestCase):

    def test_freevar(self):
        """Make sure we can compile the following kernel with freevar reference
        in arguments to shared.array
        """
        from numba import float32
        size = 1024
        nbtype = float32

        @cuda.jit('(float32[::1], intp)')
        def foo(A, i):
            """Dummy function"""
            sdata = cuda.shared.array(size, dtype=nbtype)
            A[i] = sdata[i]
        A = np.arange(2, dtype='float32')
        foo[1, 1](A, 0)