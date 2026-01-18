from unittest import TestCase
from numpy.testing import assert_equal, assert_array_equal
import numpy as np
from srsly import msgpack
def test_numpy_array_mixed(self):
    x = np.array([(1, 2, b'a', [1.0, 2.0])], np.dtype([('arg0', np.uint32), ('arg1', np.uint32), ('arg2', 'S1'), ('arg3', np.float32, (2,))]))
    x_rec = self.encode_decode(x)
    assert_array_equal(x, x_rec)
    assert_equal(x.dtype, x_rec.dtype)