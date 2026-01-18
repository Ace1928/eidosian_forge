from unittest import TestCase
from numpy.testing import assert_equal, assert_array_equal
import numpy as np
from srsly import msgpack
def test_numpy_array_noncontiguous(self):
    x = np.ones((10, 10), np.uint32)[0:5, 0:5]
    x_rec = self.encode_decode(x)
    assert_array_equal(x, x_rec)
    assert_equal(x.dtype, x_rec.dtype)