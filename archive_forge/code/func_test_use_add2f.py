import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError
def test_use_add2f(self):

    @cuda.jit('float32(float32, float32)', device=True)
    def add2f(a, b):
        return a + b

    def use_add2f(ary):
        i = cuda.grid(1)
        ary[i] = add2f(ary[i], ary[i])
    compiled = cuda.jit('void(float32[:])')(use_add2f)
    nelem = 10
    ary = np.arange(nelem, dtype=np.float32)
    exp = ary + ary
    compiled[1, nelem](ary)
    self.assertTrue(np.all(ary == exp), (ary, exp))