import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
class TestApproxEqual:

    def setup_method(self):
        self._assert_func = assert_approx_equal

    def test_simple_0d_arrays(self):
        x = np.array(1234.22)
        y = np.array(1234.23)
        self._assert_func(x, y, significant=5)
        self._assert_func(x, y, significant=6)
        assert_raises(AssertionError, lambda: self._assert_func(x, y, significant=7))

    def test_simple_items(self):
        x = 1234.22
        y = 1234.23
        self._assert_func(x, y, significant=4)
        self._assert_func(x, y, significant=5)
        self._assert_func(x, y, significant=6)
        assert_raises(AssertionError, lambda: self._assert_func(x, y, significant=7))

    def test_nan_array(self):
        anan = np.array(np.nan)
        aone = np.array(1)
        ainf = np.array(np.inf)
        self._assert_func(anan, anan)
        assert_raises(AssertionError, lambda: self._assert_func(anan, aone))
        assert_raises(AssertionError, lambda: self._assert_func(anan, ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, anan))

    def test_nan_items(self):
        anan = np.array(np.nan)
        aone = np.array(1)
        ainf = np.array(np.inf)
        self._assert_func(anan, anan)
        assert_raises(AssertionError, lambda: self._assert_func(anan, aone))
        assert_raises(AssertionError, lambda: self._assert_func(anan, ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, anan))