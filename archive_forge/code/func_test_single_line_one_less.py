from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
def test_single_line_one_less(self):
    p = poly.Polynomial([12345678, 12345678, 12345678, 12345678, 123])
    assert_equal(len(str(p)), 74)
    assert_equal(str(p), '12345678.0 + 12345678.0 x + 12345678.0 x**2 + 12345678.0 x**3 + 123.0 x**4')