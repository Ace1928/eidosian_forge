import numpy as np
from collections import namedtuple
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
def test_tuple_of_tuples(self):

    @cuda.jit
    def f(r, x):
        r[0] = len(x)
        r[1] = len(x[0])
        r[2] = len(x[1])
        r[3] = len(x[2])
        r[4] = x[1][0]
        r[5] = x[1][1]
        r[6] = x[2][0]
        r[7] = x[2][1]
        r[8] = x[2][2]
    x = ((), (5, 6), (8, 9, 10))
    r = np.ones(9, dtype=np.int64)
    f[1, 1](r, x)
    self.assertEqual(r[0], 3)
    self.assertEqual(r[1], 0)
    self.assertEqual(r[2], 2)
    self.assertEqual(r[3], 3)
    self.assertEqual(r[4], 5)
    self.assertEqual(r[5], 6)
    self.assertEqual(r[6], 8)
    self.assertEqual(r[7], 9)
    self.assertEqual(r[8], 10)