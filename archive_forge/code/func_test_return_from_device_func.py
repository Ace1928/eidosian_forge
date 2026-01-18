import numpy as np
from numba import int8, int16, int32
from numba import cuda, vectorize, njit
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.enum_usecases import (
def test_return_from_device_func(self):

    @njit
    def inner(pred):
        return Color.red if pred else Color.green

    def f(pred, out):
        out[0] = inner(pred) == Color.red
        out[1] = inner(not pred) == Color.green
    cuda_f = cuda.jit(f)
    got = np.zeros((2,), dtype=np.bool_)
    expected = got.copy()
    f(True, expected)
    cuda_f[1, 1](True, got)
    self.assertPreciseEqual(expected, got)