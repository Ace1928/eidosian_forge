import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_regression_1(self):
    a, = np.ix_(range(0))
    assert_equal(a.dtype, np.intp)
    a, = np.ix_([])
    assert_equal(a.dtype, np.intp)
    a, = np.ix_(np.array([], dtype=np.float32))
    assert_equal(a.dtype, np.float32)