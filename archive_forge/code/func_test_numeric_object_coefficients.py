from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
@pytest.mark.parametrize(('coefs', 'tgt'), ((array([Fraction(1, 2), Fraction(3, 4)], dtype=object), '1/2 + 3/4·x'), (array([1, 2, Fraction(5, 7)], dtype=object), '1 + 2·x + 5/7·x²'), (array([Decimal('1.00'), Decimal('2.2'), 3], dtype=object), '1.00 + 2.2·x + 3·x²')))
def test_numeric_object_coefficients(coefs, tgt):
    p = poly.Polynomial(coefs)
    poly.set_default_printstyle('unicode')
    assert_equal(str(p), tgt)