import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def test_strided_1d(self):
    nparr = np.empty(4)
    arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
    xx = (-2, -1, 1, 2)
    for x in xx:
        expect = nparr[::x]
        got = arr[::x]
        self.assertSameContig(got, expect)
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)