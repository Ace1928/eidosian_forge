import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestArrayConversion:

    def test_asfarray(self):
        a = asfarray(np.array([1, 2, 3]))
        assert_equal(a.__class__, np.ndarray)
        assert_(np.issubdtype(a.dtype, np.floating))
        assert_raises(TypeError, asfarray, np.array([1, 2, 3]), dtype=np.array(1.0))