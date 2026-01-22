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
class TestBitwiseUFuncs:
    bitwise_types = [np.dtype(c) for c in '?' + 'bBhHiIlLqQ' + 'O']

    def test_values(self):
        for dt in self.bitwise_types:
            zeros = np.array([0], dtype=dt)
            ones = np.array([-1]).astype(dt)
            msg = "dt = '%s'" % dt.char
            assert_equal(np.bitwise_not(zeros), ones, err_msg=msg)
            assert_equal(np.bitwise_not(ones), zeros, err_msg=msg)
            assert_equal(np.bitwise_or(zeros, zeros), zeros, err_msg=msg)
            assert_equal(np.bitwise_or(zeros, ones), ones, err_msg=msg)
            assert_equal(np.bitwise_or(ones, zeros), ones, err_msg=msg)
            assert_equal(np.bitwise_or(ones, ones), ones, err_msg=msg)
            assert_equal(np.bitwise_xor(zeros, zeros), zeros, err_msg=msg)
            assert_equal(np.bitwise_xor(zeros, ones), ones, err_msg=msg)
            assert_equal(np.bitwise_xor(ones, zeros), ones, err_msg=msg)
            assert_equal(np.bitwise_xor(ones, ones), zeros, err_msg=msg)
            assert_equal(np.bitwise_and(zeros, zeros), zeros, err_msg=msg)
            assert_equal(np.bitwise_and(zeros, ones), zeros, err_msg=msg)
            assert_equal(np.bitwise_and(ones, zeros), zeros, err_msg=msg)
            assert_equal(np.bitwise_and(ones, ones), ones, err_msg=msg)

    def test_types(self):
        for dt in self.bitwise_types:
            zeros = np.array([0], dtype=dt)
            ones = np.array([-1]).astype(dt)
            msg = "dt = '%s'" % dt.char
            assert_(np.bitwise_not(zeros).dtype == dt, msg)
            assert_(np.bitwise_or(zeros, zeros).dtype == dt, msg)
            assert_(np.bitwise_xor(zeros, zeros).dtype == dt, msg)
            assert_(np.bitwise_and(zeros, zeros).dtype == dt, msg)

    def test_identity(self):
        assert_(np.bitwise_or.identity == 0, 'bitwise_or')
        assert_(np.bitwise_xor.identity == 0, 'bitwise_xor')
        assert_(np.bitwise_and.identity == -1, 'bitwise_and')

    def test_reduction(self):
        binary_funcs = (np.bitwise_or, np.bitwise_xor, np.bitwise_and)
        for dt in self.bitwise_types:
            zeros = np.array([0], dtype=dt)
            ones = np.array([-1]).astype(dt)
            for f in binary_funcs:
                msg = "dt: '%s', f: '%s'" % (dt, f)
                assert_equal(f.reduce(zeros), zeros, err_msg=msg)
                assert_equal(f.reduce(ones), ones, err_msg=msg)
        for dt in self.bitwise_types[:-1]:
            empty = np.array([], dtype=dt)
            for f in binary_funcs:
                msg = "dt: '%s', f: '%s'" % (dt, f)
                tgt = np.array(f.identity).astype(dt)
                res = f.reduce(empty)
                assert_equal(res, tgt, err_msg=msg)
                assert_(res.dtype == tgt.dtype, msg)
        for f in binary_funcs:
            msg = "dt: '%s'" % (f,)
            empty = np.array([], dtype=object)
            tgt = f.identity
            res = f.reduce(empty)
            assert_equal(res, tgt, err_msg=msg)
        for f in binary_funcs:
            msg = "dt: '%s'" % (f,)
            btype = np.array([True], dtype=object)
            assert_(type(f.reduce(btype)) is bool, msg)