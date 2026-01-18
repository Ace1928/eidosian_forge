import numpy as np
from numpy.testing import (
import pytest
def test_poly1d_math(self):
    p = np.poly1d([1.0, 2, 4])
    q = np.poly1d([4.0, 2, 1])
    assert_equal(p / q, (np.poly1d([0.25]), np.poly1d([1.5, 3.75])))
    assert_equal(p.integ(), np.poly1d([1 / 3, 1.0, 4.0, 0.0]))
    assert_equal(p.integ(1), np.poly1d([1 / 3, 1.0, 4.0, 0.0]))
    p = np.poly1d([1.0, 2, 3])
    q = np.poly1d([3.0, 2, 1])
    assert_equal(p * q, np.poly1d([3.0, 8.0, 14.0, 8.0, 3.0]))
    assert_equal(p + q, np.poly1d([4.0, 4.0, 4.0]))
    assert_equal(p - q, np.poly1d([-2.0, 0.0, 2.0]))
    assert_equal(p ** 4, np.poly1d([1.0, 8.0, 36.0, 104.0, 214.0, 312.0, 324.0, 216.0, 81.0]))
    assert_equal(p(q), np.poly1d([9.0, 12.0, 16.0, 8.0, 6.0]))
    assert_equal(q(p), np.poly1d([3.0, 12.0, 32.0, 40.0, 34.0]))
    assert_equal(p.deriv(), np.poly1d([2.0, 2.0]))
    assert_equal(p.deriv(2), np.poly1d([2.0]))
    assert_equal(np.polydiv(np.poly1d([1, 0, -1]), np.poly1d([1, 1])), (np.poly1d([1.0, -1.0]), np.poly1d([0.0])))