import numpy as np
from collections import namedtuple
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
def test_tuple_of_tuples_and_scalars(self):

    @cuda.jit
    def f(r, x):
        r[0] = len(x)
        r[1] = len(x[0])
        r[2] = x[0][0]
        r[3] = x[0][1]
        r[4] = x[0][2]
        r[5] = x[1]
    x = ((6, 5, 4), 7)
    r = np.ones(9, dtype=np.int64)
    f[1, 1](r, x)
    self.assertEqual(r[0], 2)
    self.assertEqual(r[1], 3)
    self.assertEqual(r[2], 6)
    self.assertEqual(r[3], 5)
    self.assertEqual(r[4], 4)
    self.assertEqual(r[5], 7)