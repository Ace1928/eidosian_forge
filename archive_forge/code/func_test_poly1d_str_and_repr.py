import numpy as np
from numpy.testing import (
import pytest
def test_poly1d_str_and_repr(self):
    p = np.poly1d([1.0, 2, 3])
    assert_equal(repr(p), 'poly1d([1., 2., 3.])')
    assert_equal(str(p), '   2\n1 x + 2 x + 3')
    q = np.poly1d([3.0, 2, 1])
    assert_equal(repr(q), 'poly1d([3., 2., 1.])')
    assert_equal(str(q), '   2\n3 x + 2 x + 1')
    r = np.poly1d([1.89999 + 2j, -3j, -5.12345678, 2 + 1j])
    assert_equal(str(r), '            3      2\n(1.9 + 2j) x - 3j x - 5.123 x + (2 + 1j)')
    assert_equal(str(np.poly1d([-3, -2, -1])), '    2\n-3 x - 2 x - 1')