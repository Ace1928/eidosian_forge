from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_window():
    np.random.seed(0)
    n = 1000
    rand = np.random.standard_normal(n) + 100
    ones = np.ones(n)
    assert_array_equal(mlab.window_none(ones), ones)
    assert_array_equal(mlab.window_none(rand), rand)
    assert_array_equal(np.hanning(len(rand)) * rand, mlab.window_hanning(rand))
    assert_array_equal(np.hanning(len(ones)), mlab.window_hanning(ones))