from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
def test_set_default_printoptions():
    p = poly.Polynomial([1, 2, 3])
    c = poly.Chebyshev([1, 2, 3])
    poly.set_default_printstyle('ascii')
    assert_equal(str(p), '1.0 + 2.0 x + 3.0 x**2')
    assert_equal(str(c), '1.0 + 2.0 T_1(x) + 3.0 T_2(x)')
    poly.set_default_printstyle('unicode')
    assert_equal(str(p), '1.0 + 2.0·x + 3.0·x²')
    assert_equal(str(c), '1.0 + 2.0·T₁(x) + 3.0·T₂(x)')
    with pytest.raises(ValueError):
        poly.set_default_printstyle('invalid_input')