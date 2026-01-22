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
class TestZmapZscore:

    @pytest.mark.parametrize('x, y', [([1, 2, 3, 4], [1, 2, 3, 4]), ([1, 2, 3], [0, 1, 2, 3, 4])])
    def test_zmap(self, x, y):
        z = stats.zmap(x, y)
        expected = (x - np.mean(y)) / np.std(y)
        assert_allclose(z, expected, rtol=1e-12)

    def test_zmap_axis(self):
        x = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 2.0], [2.0, 0.0, 2.0, 0.0]])
        t1 = 1.0 / np.sqrt(2.0 / 3)
        t2 = np.sqrt(3.0) / 3
        t3 = np.sqrt(2.0)
        z0 = stats.zmap(x, x, axis=0)
        z1 = stats.zmap(x, x, axis=1)
        z0_expected = [[-t1, -t3 / 2, -t3 / 2, 0.0], [0.0, t3, -t3 / 2, t1], [t1, -t3 / 2, t3, -t1]]
        z1_expected = [[-1.0, -1.0, 1.0, 1.0], [-t2, -t2, -t2, np.sqrt(3.0)], [1.0, -1.0, 1.0, -1.0]]
        assert_array_almost_equal(z0, z0_expected)
        assert_array_almost_equal(z1, z1_expected)

    def test_zmap_ddof(self):
        x = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]])
        z = stats.zmap(x, x, axis=1, ddof=1)
        z0_expected = np.array([-0.5, -0.5, 0.5, 0.5]) / (1.0 / np.sqrt(3))
        z1_expected = np.array([-1.5, -0.5, 0.5, 1.5]) / np.sqrt(5.0 / 3)
        assert_array_almost_equal(z[0], z0_expected)
        assert_array_almost_equal(z[1], z1_expected)

    @pytest.mark.parametrize('ddof', [0, 2])
    def test_zmap_nan_policy_omit(self, ddof):
        scores = np.array([-3, -1, 2, np.nan])
        compare = np.array([-8, -3, 2, 7, 12, np.nan])
        z = stats.zmap(scores, compare, ddof=ddof, nan_policy='omit')
        assert_allclose(z, stats.zmap(scores, compare[~np.isnan(compare)], ddof=ddof))

    @pytest.mark.parametrize('ddof', [0, 2])
    def test_zmap_nan_policy_omit_with_axis(self, ddof):
        scores = np.arange(-5.0, 9.0).reshape(2, -1)
        compare = np.linspace(-8, 6, 24).reshape(2, -1)
        compare[0, 4] = np.nan
        compare[0, 6] = np.nan
        compare[1, 1] = np.nan
        z = stats.zmap(scores, compare, nan_policy='omit', axis=1, ddof=ddof)
        expected = np.array([stats.zmap(scores[0], compare[0][~np.isnan(compare[0])], ddof=ddof), stats.zmap(scores[1], compare[1][~np.isnan(compare[1])], ddof=ddof)])
        assert_allclose(z, expected, rtol=1e-14)

    def test_zmap_nan_policy_raise(self):
        scores = np.array([1, 2, 3])
        compare = np.array([-8, -3, 2, 7, 12, np.nan])
        with pytest.raises(ValueError, match='input contains nan'):
            stats.zmap(scores, compare, nan_policy='raise')

    def test_zscore(self):
        y = stats.zscore([1, 2, 3, 4])
        desired = [-1.3416407864999, -0.44721359549996, 0.44721359549996, 1.3416407864999]
        assert_array_almost_equal(desired, y, decimal=12)

    def test_zscore_axis(self):
        x = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 2.0], [2.0, 0.0, 2.0, 0.0]])
        t1 = 1.0 / np.sqrt(2.0 / 3)
        t2 = np.sqrt(3.0) / 3
        t3 = np.sqrt(2.0)
        z0 = stats.zscore(x, axis=0)
        z1 = stats.zscore(x, axis=1)
        z0_expected = [[-t1, -t3 / 2, -t3 / 2, 0.0], [0.0, t3, -t3 / 2, t1], [t1, -t3 / 2, t3, -t1]]
        z1_expected = [[-1.0, -1.0, 1.0, 1.0], [-t2, -t2, -t2, np.sqrt(3.0)], [1.0, -1.0, 1.0, -1.0]]
        assert_array_almost_equal(z0, z0_expected)
        assert_array_almost_equal(z1, z1_expected)

    def test_zscore_ddof(self):
        x = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]])
        z = stats.zscore(x, axis=1, ddof=1)
        z0_expected = np.array([-0.5, -0.5, 0.5, 0.5]) / (1.0 / np.sqrt(3))
        z1_expected = np.array([-1.5, -0.5, 0.5, 1.5]) / np.sqrt(5.0 / 3)
        assert_array_almost_equal(z[0], z0_expected)
        assert_array_almost_equal(z[1], z1_expected)

    def test_zscore_nan_propagate(self):
        x = np.array([1, 2, np.nan, 4, 5])
        z = stats.zscore(x, nan_policy='propagate')
        assert all(np.isnan(z))

    def test_zscore_nan_omit(self):
        x = np.array([1, 2, np.nan, 4, 5])
        z = stats.zscore(x, nan_policy='omit')
        expected = np.array([-1.2649110640673518, -0.6324555320336759, np.nan, 0.6324555320336759, 1.2649110640673518])
        assert_array_almost_equal(z, expected)

    def test_zscore_nan_omit_with_ddof(self):
        x = np.array([np.nan, 1.0, 3.0, 5.0, 7.0, 9.0])
        z = stats.zscore(x, ddof=1, nan_policy='omit')
        expected = np.r_[np.nan, stats.zscore(x[1:], ddof=1)]
        assert_allclose(z, expected, rtol=1e-13)

    def test_zscore_nan_raise(self):
        x = np.array([1, 2, np.nan, 4, 5])
        assert_raises(ValueError, stats.zscore, x, nan_policy='raise')

    def test_zscore_constant_input_1d(self):
        x = [-0.087] * 3
        z = stats.zscore(x)
        assert_equal(z, np.full(len(x), np.nan))

    def test_zscore_constant_input_2d(self):
        x = np.array([[10.0, 10.0, 10.0, 10.0], [10.0, 11.0, 12.0, 13.0]])
        z0 = stats.zscore(x, axis=0)
        assert_equal(z0, np.array([[np.nan, -1.0, -1.0, -1.0], [np.nan, 1.0, 1.0, 1.0]]))
        z1 = stats.zscore(x, axis=1)
        assert_equal(z1, np.array([[np.nan, np.nan, np.nan, np.nan], stats.zscore(x[1])]))
        z = stats.zscore(x, axis=None)
        assert_equal(z, stats.zscore(x.ravel()).reshape(x.shape))
        y = np.ones((3, 6))
        z = stats.zscore(y, axis=None)
        assert_equal(z, np.full(y.shape, np.nan))

    def test_zscore_constant_input_2d_nan_policy_omit(self):
        x = np.array([[10.0, 10.0, 10.0, 10.0], [10.0, 11.0, 12.0, np.nan], [10.0, 12.0, np.nan, 10.0]])
        z0 = stats.zscore(x, nan_policy='omit', axis=0)
        s = np.sqrt(3 / 2)
        s2 = np.sqrt(2)
        assert_allclose(z0, np.array([[np.nan, -s, -1.0, np.nan], [np.nan, 0, 1.0, np.nan], [np.nan, s, np.nan, np.nan]]))
        z1 = stats.zscore(x, nan_policy='omit', axis=1)
        assert_allclose(z1, np.array([[np.nan, np.nan, np.nan, np.nan], [-s, 0, s, np.nan], [-s2 / 2, s2, np.nan, -s2 / 2]]))

    def test_zscore_2d_all_nan_row(self):
        x = np.array([[np.nan, np.nan, np.nan, np.nan], [10.0, 10.0, 12.0, 12.0]])
        z = stats.zscore(x, nan_policy='omit', axis=1)
        assert_equal(z, np.array([[np.nan, np.nan, np.nan, np.nan], [-1.0, -1.0, 1.0, 1.0]]))

    def test_zscore_2d_all_nan(self):
        y = np.full((2, 3), np.nan)
        z = stats.zscore(y, nan_policy='omit', axis=None)
        assert_equal(z, y)

    @pytest.mark.parametrize('x', [np.array([]), np.zeros((3, 0, 5))])
    def test_zscore_empty_input(self, x):
        z = stats.zscore(x)
        assert_equal(z, x)

    def test_gzscore_normal_array(self):
        z = stats.gzscore([1, 2, 3, 4])
        desired = [-1.526072095151, -0.194700599824, 0.584101799472, 1.136670895503]
        assert_allclose(desired, z)

    def test_gzscore_masked_array(self):
        x = np.array([1, 2, -1, 3, 4])
        mx = np.ma.masked_array(x, mask=[0, 0, 1, 0, 0])
        z = stats.gzscore(mx)
        desired = [-1.526072095151, -0.194700599824, np.inf, 0.584101799472, 1.136670895503]
        assert_allclose(desired, z)

    def test_zscore_masked_element_0_gh19039(self):
        rng = np.random.default_rng(8675309)
        x = rng.standard_normal(10)
        mask = np.zeros_like(x)
        y = np.ma.masked_array(x, mask)
        y.mask[0] = True
        ref = stats.zscore(x[1:])
        assert not np.any(np.isnan(ref))
        res = stats.zscore(y)
        assert_allclose(res[1:], ref)
        res = stats.zscore(y, axis=None)
        assert_allclose(res[1:], ref)
        y[1:] = y[1]
        res = stats.zscore(y)
        assert_equal(res[1:], np.nan)
        res = stats.zscore(y, axis=None)
        assert_equal(res[1:], np.nan)