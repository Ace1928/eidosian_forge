import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
def test_tmean(self):
    y = stats.tmean(X, (2, 8), (True, True))
    assert_approx_equal(y, 5.0, significant=self.dprec)
    y1 = stats.tmean(X, limits=(2, 8), inclusive=(False, False))
    y2 = stats.tmean(X, limits=None)
    assert_approx_equal(y1, y2, significant=self.dprec)
    x_2d = arange(63, dtype=float64).reshape(9, 7)
    y = stats.tmean(x_2d, axis=None)
    assert_approx_equal(y, x_2d.mean(), significant=self.dprec)
    y = stats.tmean(x_2d, axis=0)
    assert_array_almost_equal(y, x_2d.mean(axis=0), decimal=8)
    y = stats.tmean(x_2d, axis=1)
    assert_array_almost_equal(y, x_2d.mean(axis=1), decimal=8)
    y = stats.tmean(x_2d, limits=(2, 61), axis=None)
    assert_approx_equal(y, 31.5, significant=self.dprec)
    y = stats.tmean(x_2d, limits=(2, 21), axis=0)
    y_true = [14, 11.5, 9, 10, 11, 12, 13]
    assert_array_almost_equal(y, y_true, decimal=8)
    y = stats.tmean(x_2d, limits=(2, 21), inclusive=(True, False), axis=0)
    y_true = [10.5, 11.5, 9, 10, 11, 12, 13]
    assert_array_almost_equal(y, y_true, decimal=8)
    x_2d_with_nan = np.array(x_2d)
    x_2d_with_nan[-1, -3:] = np.nan
    y = stats.tmean(x_2d_with_nan, limits=(1, 13), axis=0)
    y_true = [7, 4.5, 5.5, 6.5, np.nan, np.nan, np.nan]
    assert_array_almost_equal(y, y_true, decimal=8)
    with suppress_warnings() as sup:
        sup.record(RuntimeWarning, 'Mean of empty slice')
        y = stats.tmean(x_2d, limits=(2, 21), axis=1)
        y_true = [4, 10, 17, 21, np.nan, np.nan, np.nan, np.nan, np.nan]
        assert_array_almost_equal(y, y_true, decimal=8)
        y = stats.tmean(x_2d, limits=(2, 21), inclusive=(False, True), axis=1)
        y_true = [4.5, 10, 17, 21, np.nan, np.nan, np.nan, np.nan, np.nan]
        assert_array_almost_equal(y, y_true, decimal=8)