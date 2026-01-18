import numpy as np
from numba import int8, int16, int32
from numba import cuda, vectorize, njit
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.enum_usecases import (
@skip_on_cudasim('ufuncs are unsupported on simulator.')
def test_vectorize(self):

    def f(x):
        if x != RequestError.not_found:
            return RequestError['internal_error']
        else:
            return RequestError.dummy
    cuda_func = vectorize('int64(int64)', target='cuda')(f)
    arr = np.array([2, 404, 500, 404], dtype=np.int64)
    expected = np.array([f(x) for x in arr], dtype=np.int64)
    got = cuda_func(arr)
    self.assertPreciseEqual(expected, got)