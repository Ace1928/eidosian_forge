import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
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