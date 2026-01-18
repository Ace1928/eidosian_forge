from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
def test_switch_to_exp(self):
    for i, s in enumerate(SWITCH_TO_EXP):
        with printoptions(precision=i):
            p = poly.Polynomial([1.23456789 * 10 ** (-i) for i in range(i // 2 + 3)])
            assert str(p).replace('\n', ' ') == s