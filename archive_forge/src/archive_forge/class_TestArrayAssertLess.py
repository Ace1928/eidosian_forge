import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
class TestArrayAssertLess:

    def setup_method(self):
        self._assert_func = assert_array_less

    def test_simple_arrays(self):
        x = np.array([1.1, 2.2])
        y = np.array([1.2, 2.3])
        self._assert_func(x, y)
        assert_raises(AssertionError, lambda: self._assert_func(y, x))
        y = np.array([1.0, 2.3])
        assert_raises(AssertionError, lambda: self._assert_func(x, y))
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

    def test_rank2(self):
        x = np.array([[1.1, 2.2], [3.3, 4.4]])
        y = np.array([[1.2, 2.3], [3.4, 4.5]])
        self._assert_func(x, y)
        assert_raises(AssertionError, lambda: self._assert_func(y, x))
        y = np.array([[1.0, 2.3], [3.4, 4.5]])
        assert_raises(AssertionError, lambda: self._assert_func(x, y))
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

    def test_rank3(self):
        x = np.ones(shape=(2, 2, 2))
        y = np.ones(shape=(2, 2, 2)) + 1
        self._assert_func(x, y)
        assert_raises(AssertionError, lambda: self._assert_func(y, x))
        y[0, 0, 0] = 0
        assert_raises(AssertionError, lambda: self._assert_func(x, y))
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

    def test_simple_items(self):
        x = 1.1
        y = 2.2
        self._assert_func(x, y)
        assert_raises(AssertionError, lambda: self._assert_func(y, x))
        y = np.array([2.2, 3.3])
        self._assert_func(x, y)
        assert_raises(AssertionError, lambda: self._assert_func(y, x))
        y = np.array([1.0, 3.3])
        assert_raises(AssertionError, lambda: self._assert_func(x, y))

    def test_nan_noncompare(self):
        anan = np.array(np.nan)
        aone = np.array(1)
        ainf = np.array(np.inf)
        self._assert_func(anan, anan)
        assert_raises(AssertionError, lambda: self._assert_func(aone, anan))
        assert_raises(AssertionError, lambda: self._assert_func(anan, aone))
        assert_raises(AssertionError, lambda: self._assert_func(anan, ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, anan))

    def test_nan_noncompare_array(self):
        x = np.array([1.1, 2.2, 3.3])
        anan = np.array(np.nan)
        assert_raises(AssertionError, lambda: self._assert_func(x, anan))
        assert_raises(AssertionError, lambda: self._assert_func(anan, x))
        x = np.array([1.1, 2.2, np.nan])
        assert_raises(AssertionError, lambda: self._assert_func(x, anan))
        assert_raises(AssertionError, lambda: self._assert_func(anan, x))
        y = np.array([1.0, 2.0, np.nan])
        self._assert_func(y, x)
        assert_raises(AssertionError, lambda: self._assert_func(x, y))

    def test_inf_compare(self):
        aone = np.array(1)
        ainf = np.array(np.inf)
        self._assert_func(aone, ainf)
        self._assert_func(-ainf, aone)
        self._assert_func(-ainf, ainf)
        assert_raises(AssertionError, lambda: self._assert_func(ainf, aone))
        assert_raises(AssertionError, lambda: self._assert_func(aone, -ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, -ainf))
        assert_raises(AssertionError, lambda: self._assert_func(-ainf, -ainf))

    def test_inf_compare_array(self):
        x = np.array([1.1, 2.2, np.inf])
        ainf = np.array(np.inf)
        assert_raises(AssertionError, lambda: self._assert_func(x, ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, x))
        assert_raises(AssertionError, lambda: self._assert_func(x, -ainf))
        assert_raises(AssertionError, lambda: self._assert_func(-x, -ainf))
        assert_raises(AssertionError, lambda: self._assert_func(-ainf, -x))
        self._assert_func(-ainf, x)