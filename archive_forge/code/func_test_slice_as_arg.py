import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
def test_slice_as_arg(self):
    global cufoo
    cufoo = cuda.jit('void(int32[:], int32[:])', device=True)(foo)
    cucopy = cuda.jit('void(int32[:,:], int32[:,:])')(copy)
    inp = np.arange(100, dtype=np.int32).reshape(10, 10)
    out = np.zeros_like(inp)
    cucopy[1, 10](inp, out)