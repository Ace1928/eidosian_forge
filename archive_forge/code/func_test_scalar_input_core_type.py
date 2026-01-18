import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
def test_scalar_input_core_type(self):

    def pyfunc(inp, n, out):
        for i in range(inp.size):
            out[i] = n * (inp[i] + 1)
    my_gufunc = guvectorize(['int32[:], int32, int32[:]'], '(n),()->(n)', target=self.target)(pyfunc)
    arr = np.arange(10).astype(np.int32)
    got = my_gufunc(arr, 2)
    expected = np.zeros_like(got)
    pyfunc(arr, 2, expected)
    np.testing.assert_equal(got, expected)
    arr = np.arange(20).astype(np.int32).reshape(10, 2)
    got = my_gufunc(arr, 2)
    expected = np.zeros_like(got)
    for ax in range(expected.shape[0]):
        pyfunc(arr[ax], 2, expected[ax])
    np.testing.assert_equal(got, expected)