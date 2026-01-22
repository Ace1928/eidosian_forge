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
class TestArctan2SpecialValues:

    def test_one_one(self):
        assert_almost_equal(ncu.arctan2(1, 1), 0.25 * np.pi)
        assert_almost_equal(ncu.arctan2(-1, 1), -0.25 * np.pi)
        assert_almost_equal(ncu.arctan2(1, -1), 0.75 * np.pi)

    def test_zero_nzero(self):
        assert_almost_equal(ncu.arctan2(np.PZERO, np.NZERO), np.pi)
        assert_almost_equal(ncu.arctan2(np.NZERO, np.NZERO), -np.pi)

    def test_zero_pzero(self):
        assert_arctan2_ispzero(np.PZERO, np.PZERO)
        assert_arctan2_isnzero(np.NZERO, np.PZERO)

    def test_zero_negative(self):
        assert_almost_equal(ncu.arctan2(np.PZERO, -1), np.pi)
        assert_almost_equal(ncu.arctan2(np.NZERO, -1), -np.pi)

    def test_zero_positive(self):
        assert_arctan2_ispzero(np.PZERO, 1)
        assert_arctan2_isnzero(np.NZERO, 1)

    def test_positive_zero(self):
        assert_almost_equal(ncu.arctan2(1, np.PZERO), 0.5 * np.pi)
        assert_almost_equal(ncu.arctan2(1, np.NZERO), 0.5 * np.pi)

    def test_negative_zero(self):
        assert_almost_equal(ncu.arctan2(-1, np.PZERO), -0.5 * np.pi)
        assert_almost_equal(ncu.arctan2(-1, np.NZERO), -0.5 * np.pi)

    def test_any_ninf(self):
        assert_almost_equal(ncu.arctan2(1, np.NINF), np.pi)
        assert_almost_equal(ncu.arctan2(-1, np.NINF), -np.pi)

    def test_any_pinf(self):
        assert_arctan2_ispzero(1, np.inf)
        assert_arctan2_isnzero(-1, np.inf)

    def test_inf_any(self):
        assert_almost_equal(ncu.arctan2(np.inf, 1), 0.5 * np.pi)
        assert_almost_equal(ncu.arctan2(-np.inf, 1), -0.5 * np.pi)

    def test_inf_ninf(self):
        assert_almost_equal(ncu.arctan2(np.inf, -np.inf), 0.75 * np.pi)
        assert_almost_equal(ncu.arctan2(-np.inf, -np.inf), -0.75 * np.pi)

    def test_inf_pinf(self):
        assert_almost_equal(ncu.arctan2(np.inf, np.inf), 0.25 * np.pi)
        assert_almost_equal(ncu.arctan2(-np.inf, np.inf), -0.25 * np.pi)

    def test_nan_any(self):
        assert_arctan2_isnan(np.nan, np.inf)
        assert_arctan2_isnan(np.inf, np.nan)
        assert_arctan2_isnan(np.nan, np.nan)