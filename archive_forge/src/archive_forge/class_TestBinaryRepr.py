import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
class TestBinaryRepr:

    def test_zero(self):
        assert_equal(np.binary_repr(0), '0')

    def test_positive(self):
        assert_equal(np.binary_repr(10), '1010')
        assert_equal(np.binary_repr(12522), '11000011101010')
        assert_equal(np.binary_repr(10736848), '101000111101010011010000')

    def test_negative(self):
        assert_equal(np.binary_repr(-1), '-1')
        assert_equal(np.binary_repr(-10), '-1010')
        assert_equal(np.binary_repr(-12522), '-11000011101010')
        assert_equal(np.binary_repr(-10736848), '-101000111101010011010000')

    def test_sufficient_width(self):
        assert_equal(np.binary_repr(0, width=5), '00000')
        assert_equal(np.binary_repr(10, width=7), '0001010')
        assert_equal(np.binary_repr(-5, width=7), '1111011')

    def test_neg_width_boundaries(self):
        assert_equal(np.binary_repr(-128, width=8), '10000000')
        for width in range(1, 11):
            num = -2 ** (width - 1)
            exp = '1' + (width - 1) * '0'
            assert_equal(np.binary_repr(num, width=width), exp)

    def test_large_neg_int64(self):
        assert_equal(np.binary_repr(np.int64(-2 ** 62), width=64), '11' + '0' * 62)