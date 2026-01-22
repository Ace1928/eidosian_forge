import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
class TestGUVectorizeScalar(TestCase):
    """
    Nothing keeps user from out-of-bound memory access
    """
    target = 'cpu'

    def test_scalar_output(self):
        """
        Note that scalar output is a 0-dimension array that acts as
        a pointer to the output location.
        """

        @guvectorize(['void(int32[:], int32[:])'], '(n)->()', target=self.target, nopython=True)
        def sum_row(inp, out):
            tmp = 0.0
            for i in range(inp.shape[0]):
                tmp += inp[i]
            out[()] = tmp
        inp = np.arange(30000, dtype=np.int32).reshape(10000, 3)
        out = sum_row(inp)
        for i in range(inp.shape[0]):
            self.assertEqual(out[i], inp[i].sum())

    def test_scalar_input(self):

        @guvectorize(['int32[:], int32[:], int32[:]'], '(n),()->(n)', target=self.target, nopython=True)
        def foo(inp, n, out):
            for i in range(inp.shape[0]):
                out[i] = inp[i] * n[0]
        inp = np.arange(3 * 10, dtype=np.int32).reshape(10, 3)
        out = foo(inp, 2)
        self.assertPreciseEqual(inp * 2, out)

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

    def test_scalar_input_core_type_error(self):
        with self.assertRaises(TypeError) as raises:

            @guvectorize(['int32[:], int32, int32[:]'], '(n),(n)->(n)', target=self.target)
            def pyfunc(a, b, c):
                pass
        self.assertEqual('scalar type int32 given for non scalar argument #2', str(raises.exception))

    def test_ndim_mismatch(self):
        with self.assertRaises(TypeError) as raises:

            @guvectorize(['int32[:], int32[:]'], '(m,n)->(n)', target=self.target)
            def pyfunc(a, b):
                pass
        self.assertEqual('type and shape signature mismatch for arg #1', str(raises.exception))