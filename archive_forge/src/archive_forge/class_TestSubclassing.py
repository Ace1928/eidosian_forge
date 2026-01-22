import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import (assert_, assert_equal, assert_raises,
from numpy.ma.core import (masked_array, masked_values, masked, allequal,
from numpy.ma.extras import mr_
from numpy.compat import pickle
class TestSubclassing:

    def setup_method(self):
        x = np.arange(5, dtype='float')
        mx = MMatrix(x, mask=[0, 1, 0, 0, 0])
        self.data = (x, mx)

    def test_maskedarray_subclassing(self):
        x, mx = self.data
        assert_(isinstance(mx._data, np.matrix))

    def test_masked_unary_operations(self):
        x, mx = self.data
        with np.errstate(divide='ignore'):
            assert_(isinstance(log(mx), MMatrix))
            assert_equal(log(x), np.log(x))

    def test_masked_binary_operations(self):
        x, mx = self.data
        assert_(isinstance(add(mx, mx), MMatrix))
        assert_(isinstance(add(mx, x), MMatrix))
        assert_equal(add(mx, x), mx + x)
        assert_(isinstance(add(mx, mx)._data, np.matrix))
        with assert_warns(DeprecationWarning):
            assert_(isinstance(add.outer(mx, mx), MMatrix))
        assert_(isinstance(hypot(mx, mx), MMatrix))
        assert_(isinstance(hypot(mx, x), MMatrix))

    def test_masked_binary_operations2(self):
        x, mx = self.data
        xmx = masked_array(mx.data.__array__(), mask=mx.mask)
        assert_(isinstance(divide(mx, mx), MMatrix))
        assert_(isinstance(divide(mx, x), MMatrix))
        assert_equal(divide(mx, mx), divide(xmx, xmx))