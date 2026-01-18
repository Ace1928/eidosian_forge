import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def test_slice2_2d(self):
    nparr = np.empty((4, 5))
    arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
    xx = (-2, 0, 2)
    for s, t, u, v in itertools.product(xx, xx, xx, xx):
        expect = nparr[s:t, u:v]
        got = arr[s:t, u:v]
        self.assertSameContig(got, expect)
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)
    for x, y in itertools.product(xx, xx):
        expect = nparr[s:t, u:v]
        got = arr[s:t, u:v]
        self.assertSameContig(got, expect)
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)