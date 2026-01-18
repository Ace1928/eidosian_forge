import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def test_reshape_3d3d(self):
    nparr = np.empty((3, 4, 5))
    arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
    expect = nparr.reshape(5, 3, 4)
    got = arr.reshape(5, 3, 4)[0]
    self.assertEqual(got.shape, expect.shape)
    self.assertEqual(got.strides, expect.strides)