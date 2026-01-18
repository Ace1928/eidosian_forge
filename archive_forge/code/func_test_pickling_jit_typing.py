import pickle
import numpy as np
from numba import cuda, vectorize
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_pickling_jit_typing(self):

    @cuda.jit(device=True)
    def inner(a):
        return a + 1

    @cuda.jit('void(intp[:])')
    def foo(arr):
        arr[0] = inner(arr[0])
    self.check_call(foo)