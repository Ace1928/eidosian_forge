import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises as assert_raises
from scipy.signal._arraytools import (axis_slice, axis_reverse,
def test_odd_ext(self):
    a = np.array([[1, 2, 3, 4, 5], [9, 8, 7, 6, 5]])
    odd = odd_ext(a, 2, axis=1)
    expected = np.array([[-1, 0, 1, 2, 3, 4, 5, 6, 7], [11, 10, 9, 8, 7, 6, 5, 4, 3]])
    assert_array_equal(odd, expected)
    odd = odd_ext(a, 1, axis=0)
    expected = np.array([[-7, -4, -1, 2, 5], [1, 2, 3, 4, 5], [9, 8, 7, 6, 5], [17, 14, 11, 8, 5]])
    assert_array_equal(odd, expected)
    assert_raises(ValueError, odd_ext, a, 2, axis=0)
    assert_raises(ValueError, odd_ext, a, 5, axis=1)