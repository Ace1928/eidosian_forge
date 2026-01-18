import numpy as np
from numpy.testing import (
import pytest
def test_poly_coeffs_mutable(self):
    """ Coefficients should be modifiable """
    p = np.poly1d([1, 2, 3])
    p.coeffs += 1
    assert_equal(p.coeffs, [2, 3, 4])
    p.coeffs[2] += 10
    assert_equal(p.coeffs, [2, 3, 14])
    assert_raises(AttributeError, setattr, p, 'coeffs', np.array(1))