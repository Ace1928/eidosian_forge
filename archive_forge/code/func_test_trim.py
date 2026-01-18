import warnings
import platform
import numpy as np
from numpy import nan
import numpy.ma as ma
from numpy.ma import masked, nomask
import scipy.stats.mstats as mstats
from scipy import stats
from .common_tests import check_named_results
import pytest
from pytest import raises as assert_raises
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
from numpy.testing import suppress_warnings
from scipy.stats import _mstats_basic
def test_trim(self):
    a = ma.arange(10)
    assert_equal(mstats.trim(a), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    a = ma.arange(10)
    assert_equal(mstats.trim(a, (2, 8)), [None, None, 2, 3, 4, 5, 6, 7, 8, None])
    a = ma.arange(10)
    assert_equal(mstats.trim(a, limits=(2, 8), inclusive=(False, False)), [None, None, None, 3, 4, 5, 6, 7, None, None])
    a = ma.arange(10)
    assert_equal(mstats.trim(a, limits=(0.1, 0.2), relative=True), [None, 1, 2, 3, 4, 5, 6, 7, None, None])
    a = ma.arange(12)
    a[[0, -1]] = a[5] = masked
    assert_equal(mstats.trim(a, (2, 8)), [None, None, 2, 3, 4, None, 6, 7, 8, None, None, None])
    x = ma.arange(100).reshape(10, 10)
    expected = [1] * 10 + [0] * 70 + [1] * 20
    trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=None)
    assert_equal(trimx._mask.ravel(), expected)
    trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=0)
    assert_equal(trimx._mask.ravel(), expected)
    trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=-1)
    assert_equal(trimx._mask.T.ravel(), expected)
    x = ma.arange(110).reshape(11, 10)
    x[1] = masked
    expected = [1] * 20 + [0] * 70 + [1] * 20
    trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=None)
    assert_equal(trimx._mask.ravel(), expected)
    trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=0)
    assert_equal(trimx._mask.ravel(), expected)
    trimx = mstats.trim(x.T, (0.1, 0.2), relative=True, axis=-1)
    assert_equal(trimx.T._mask.ravel(), expected)