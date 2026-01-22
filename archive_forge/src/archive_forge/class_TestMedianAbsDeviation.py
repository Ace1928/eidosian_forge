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
class TestMedianAbsDeviation:

    def setup_class(self):
        self.dat_nan = np.array([2.2, 2.2, 2.4, 2.4, 2.5, 2.7, 2.8, 2.9, 3.03, 3.03, 3.1, 3.37, 3.4, 3.4, 3.4, 3.5, 3.6, 3.7, 3.7, 3.7, 3.7, 3.77, 5.28, np.nan])
        self.dat = np.array([2.2, 2.2, 2.4, 2.4, 2.5, 2.7, 2.8, 2.9, 3.03, 3.03, 3.1, 3.37, 3.4, 3.4, 3.4, 3.5, 3.6, 3.7, 3.7, 3.7, 3.7, 3.77, 5.28, 28.95])

    def test_median_abs_deviation(self):
        assert_almost_equal(stats.median_abs_deviation(self.dat, axis=None), 0.355)
        dat = self.dat.reshape(6, 4)
        mad = stats.median_abs_deviation(dat, axis=0)
        mad_expected = np.asarray([0.435, 0.5, 0.45, 0.4])
        assert_array_almost_equal(mad, mad_expected)

    def test_mad_nan_omit(self):
        mad = stats.median_abs_deviation(self.dat_nan, nan_policy='omit')
        assert_almost_equal(mad, 0.34)

    def test_axis_and_nan(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0, np.nan], [1.0, 4.0, 5.0, 8.0, 9.0]])
        mad = stats.median_abs_deviation(x, axis=1)
        assert_equal(mad, np.array([np.nan, 3.0]))

    def test_nan_policy_omit_with_inf(sef):
        z = np.array([1, 3, 4, 6, 99, np.nan, np.inf])
        mad = stats.median_abs_deviation(z, nan_policy='omit')
        assert_equal(mad, 3.0)

    @pytest.mark.parametrize('axis', [0, 1, 2, None])
    def test_size_zero_with_axis(self, axis):
        x = np.zeros((3, 0, 4))
        mad = stats.median_abs_deviation(x, axis=axis)
        assert_equal(mad, np.full_like(x.sum(axis=axis), fill_value=np.nan))

    @pytest.mark.parametrize('nan_policy, expected', [('omit', np.array([np.nan, 1.5, 1.5])), ('propagate', np.array([np.nan, np.nan, 1.5]))])
    def test_nan_policy_with_axis(self, nan_policy, expected):
        x = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [1, 5, 3, 6, np.nan, np.nan], [5, 6, 7, 9, 9, 10]])
        mad = stats.median_abs_deviation(x, nan_policy=nan_policy, axis=1)
        assert_equal(mad, expected)

    @pytest.mark.parametrize('axis, expected', [(1, [2.5, 2.0, 12.0]), (None, 4.5)])
    def test_center_mean_with_nan(self, axis, expected):
        x = np.array([[1, 2, 4, 9, np.nan], [0, 1, 1, 1, 12], [-10, -10, -10, 20, 20]])
        mad = stats.median_abs_deviation(x, center=np.mean, nan_policy='omit', axis=axis)
        assert_allclose(mad, expected, rtol=1e-15, atol=1e-15)

    def test_center_not_callable(self):
        with pytest.raises(TypeError, match='callable'):
            stats.median_abs_deviation([1, 2, 3, 5], center=99)