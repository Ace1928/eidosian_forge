import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
class TestRationalFunctions:

    def test_lcm(self):
        self._test_lcm_inner(np.int16)
        self._test_lcm_inner(np.uint16)

    def test_lcm_object(self):
        self._test_lcm_inner(np.object_)

    def test_gcd(self):
        self._test_gcd_inner(np.int16)
        self._test_lcm_inner(np.uint16)

    def test_gcd_object(self):
        self._test_gcd_inner(np.object_)

    def _test_lcm_inner(self, dtype):
        a = np.array([12, 120], dtype=dtype)
        b = np.array([20, 200], dtype=dtype)
        assert_equal(np.lcm(a, b), [60, 600])
        if not issubclass(dtype, np.unsignedinteger):
            a = np.array([12, -12, 12, -12], dtype=dtype)
            b = np.array([20, 20, -20, -20], dtype=dtype)
            assert_equal(np.lcm(a, b), [60] * 4)
        a = np.array([3, 12, 20], dtype=dtype)
        assert_equal(np.lcm.reduce([3, 12, 20]), 60)
        a = np.arange(6).astype(dtype)
        b = 20
        assert_equal(np.lcm(a, b), [0, 20, 20, 60, 20, 20])

    def _test_gcd_inner(self, dtype):
        a = np.array([12, 120], dtype=dtype)
        b = np.array([20, 200], dtype=dtype)
        assert_equal(np.gcd(a, b), [4, 40])
        if not issubclass(dtype, np.unsignedinteger):
            a = np.array([12, -12, 12, -12], dtype=dtype)
            b = np.array([20, 20, -20, -20], dtype=dtype)
            assert_equal(np.gcd(a, b), [4] * 4)
        a = np.array([15, 25, 35], dtype=dtype)
        assert_equal(np.gcd.reduce(a), 5)
        a = np.arange(6).astype(dtype)
        b = 20
        assert_equal(np.gcd(a, b), [20, 1, 2, 1, 4, 5])

    def test_lcm_overflow(self):
        big = np.int32(np.iinfo(np.int32).max // 11)
        a = 2 * big
        b = 5 * big
        assert_equal(np.lcm(a, b), 10 * big)

    def test_gcd_overflow(self):
        for dtype in (np.int32, np.int64):
            a = dtype(np.iinfo(dtype).min)
            q = -(a // 4)
            assert_equal(np.gcd(a, q * 3), q)
            assert_equal(np.gcd(a, -q * 3), q)

    def test_decimal(self):
        from decimal import Decimal
        a = np.array([1, 1, -1, -1]) * Decimal('0.20')
        b = np.array([1, -1, 1, -1]) * Decimal('0.12')
        assert_equal(np.gcd(a, b), 4 * [Decimal('0.04')])
        assert_equal(np.lcm(a, b), 4 * [Decimal('0.60')])

    def test_float(self):
        assert_raises(TypeError, np.gcd, 0.3, 0.4)
        assert_raises(TypeError, np.lcm, 0.3, 0.4)

    def test_builtin_long(self):
        assert_equal(np.array(2 ** 200).item(), 2 ** 200)
        a = np.array(2 ** 100 * 3 ** 5)
        b = np.array([2 ** 100 * 5 ** 7, 2 ** 50 * 3 ** 10])
        assert_equal(np.gcd(a, b), [2 ** 100, 2 ** 50 * 3 ** 5])
        assert_equal(np.lcm(a, b), [2 ** 100 * 3 ** 5 * 5 ** 7, 2 ** 100 * 3 ** 10])
        assert_equal(np.gcd(2 ** 100, 3 ** 100), 1)