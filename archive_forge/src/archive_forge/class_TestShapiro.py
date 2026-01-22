import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
class TestShapiro:

    def test_basic(self):
        x1 = [0.11, 7.87, 4.61, 10.14, 7.95, 3.14, 0.46, 4.43, 0.21, 4.75, 0.71, 1.52, 3.24, 0.93, 0.42, 4.97, 9.53, 4.55, 0.47, 6.66]
        w, pw = stats.shapiro(x1)
        shapiro_test = stats.shapiro(x1)
        assert_almost_equal(w, 0.9004729986190796, decimal=6)
        assert_almost_equal(shapiro_test.statistic, 0.9004729986190796, decimal=6)
        assert_almost_equal(pw, 0.04208974540233612, decimal=6)
        assert_almost_equal(shapiro_test.pvalue, 0.04208974540233612, decimal=6)
        x2 = [1.36, 1.14, 2.92, 2.55, 1.46, 1.06, 5.27, -1.11, 3.48, 1.1, 0.88, -0.51, 1.46, 0.52, 6.2, 1.69, 0.08, 3.67, 2.81, 3.49]
        w, pw = stats.shapiro(x2)
        shapiro_test = stats.shapiro(x2)
        assert_almost_equal(w, 0.959027, decimal=6)
        assert_almost_equal(shapiro_test.statistic, 0.959027, decimal=6)
        assert_almost_equal(pw, 0.5246, decimal=3)
        assert_almost_equal(shapiro_test.pvalue, 0.5246, decimal=3)
        x3 = stats.norm.rvs(loc=5, scale=3, size=100, random_state=12345678)
        w, pw = stats.shapiro(x3)
        shapiro_test = stats.shapiro(x3)
        assert_almost_equal(w, 0.9772805571556091, decimal=6)
        assert_almost_equal(shapiro_test.statistic, 0.9772805571556091, decimal=6)
        assert_almost_equal(pw, 0.08144091814756393, decimal=3)
        assert_almost_equal(shapiro_test.pvalue, 0.08144091814756393, decimal=3)
        x4 = [0.139, 0.157, 0.175, 0.256, 0.344, 0.413, 0.503, 0.577, 0.614, 0.655, 0.954, 1.392, 1.557, 1.648, 1.69, 1.994, 2.174, 2.206, 3.245, 3.51, 3.571, 4.354, 4.98, 6.084, 8.351]
        W_expected = 0.83467
        p_expected = 0.000914
        w, pw = stats.shapiro(x4)
        shapiro_test = stats.shapiro(x4)
        assert_almost_equal(w, W_expected, decimal=4)
        assert_almost_equal(shapiro_test.statistic, W_expected, decimal=4)
        assert_almost_equal(pw, p_expected, decimal=5)
        assert_almost_equal(shapiro_test.pvalue, p_expected, decimal=5)

    def test_2d(self):
        x1 = [[0.11, 7.87, 4.61, 10.14, 7.95, 3.14, 0.46, 4.43, 0.21, 4.75], [0.71, 1.52, 3.24, 0.93, 0.42, 4.97, 9.53, 4.55, 0.47, 6.66]]
        w, pw = stats.shapiro(x1)
        shapiro_test = stats.shapiro(x1)
        assert_almost_equal(w, 0.9004729986190796, decimal=6)
        assert_almost_equal(shapiro_test.statistic, 0.9004729986190796, decimal=6)
        assert_almost_equal(pw, 0.04208974540233612, decimal=6)
        assert_almost_equal(shapiro_test.pvalue, 0.04208974540233612, decimal=6)
        x2 = [[1.36, 1.14, 2.92, 2.55, 1.46, 1.06, 5.27, -1.11, 3.48, 1.1], [0.88, -0.51, 1.46, 0.52, 6.2, 1.69, 0.08, 3.67, 2.81, 3.49]]
        w, pw = stats.shapiro(x2)
        shapiro_test = stats.shapiro(x2)
        assert_almost_equal(w, 0.959027, decimal=6)
        assert_almost_equal(shapiro_test.statistic, 0.959027, decimal=6)
        assert_almost_equal(pw, 0.5246, decimal=3)
        assert_almost_equal(shapiro_test.pvalue, 0.5246, decimal=3)

    def test_empty_input(self):
        assert_raises(ValueError, stats.shapiro, [])
        assert_raises(ValueError, stats.shapiro, [[], [], []])

    def test_not_enough_values(self):
        assert_raises(ValueError, stats.shapiro, [1, 2])
        assert_raises(ValueError, stats.shapiro, np.array([[], [2]], dtype=object))

    def test_bad_arg(self):
        x = [1]
        assert_raises(ValueError, stats.shapiro, x)

    def test_nan_input(self):
        x = np.arange(10.0)
        x[9] = np.nan
        w, pw = stats.shapiro(x)
        shapiro_test = stats.shapiro(x)
        assert_equal(w, np.nan)
        assert_equal(shapiro_test.statistic, np.nan)
        assert_almost_equal(pw, 1.0)
        assert_almost_equal(shapiro_test.pvalue, 1.0)

    def test_gh14462(self):
        trans_val, maxlog = stats.boxcox([122500, 474400, 110400])
        res = stats.shapiro(trans_val)
        ref = (0.86468431705371, 0.2805581751566)
        assert_allclose(res, ref, rtol=1e-05)

    def test_length_3_gh18322(self):
        res = stats.shapiro([0.6931471805599453, 0.0, 0.0])
        assert res.pvalue >= 0
        x = [-0.7746653110021126, -0.4344432067942129, 1.8157053280290931]
        res = stats.shapiro(x)
        assert_allclose(res.statistic, 0.84658770645509)
        assert_allclose(res.pvalue, 0.2313666489882, rtol=1e-06)