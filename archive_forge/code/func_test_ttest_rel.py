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
def test_ttest_rel():
    tr, pr = (0.8124859138916569, 0.41846234511362157)
    tpr = ([tr, -tr], [pr, pr])
    rvs1 = np.linspace(1, 100, 100)
    rvs2 = np.linspace(1.01, 99.989, 100)
    rvs1_2D = np.array([np.linspace(1, 100, 100), np.linspace(1.01, 99.989, 100)])
    rvs2_2D = np.array([np.linspace(1.01, 99.989, 100), np.linspace(1, 100, 100)])
    t, p = stats.ttest_rel(rvs1, rvs2, axis=0)
    assert_array_almost_equal([t, p], (tr, pr))
    t, p = stats.ttest_rel(rvs1_2D.T, rvs2_2D.T, axis=0)
    assert_array_almost_equal([t, p], tpr)
    t, p = stats.ttest_rel(rvs1_2D, rvs2_2D, axis=1)
    assert_array_almost_equal([t, p], tpr)
    with suppress_warnings() as sup, np.errstate(invalid='ignore', divide='ignore'):
        sup.filter(RuntimeWarning, 'Degrees of freedom <= 0 for slice')
        t, p = stats.ttest_rel(4.0, 3.0)
    assert_(np.isnan(t))
    assert_(np.isnan(p))
    attributes = ('statistic', 'pvalue')
    res = stats.ttest_rel(rvs1, rvs2, axis=0)
    check_named_results(res, attributes)
    rvs1_3D = np.dstack([rvs1_2D, rvs1_2D, rvs1_2D])
    rvs2_3D = np.dstack([rvs2_2D, rvs2_2D, rvs2_2D])
    t, p = stats.ttest_rel(rvs1_3D, rvs2_3D, axis=1)
    assert_array_almost_equal(np.abs(t), tr)
    assert_array_almost_equal(np.abs(p), pr)
    assert_equal(t.shape, (2, 3))
    t, p = stats.ttest_rel(np.moveaxis(rvs1_3D, 2, 0), np.moveaxis(rvs2_3D, 2, 0), axis=2)
    assert_array_almost_equal(np.abs(t), tr)
    assert_array_almost_equal(np.abs(p), pr)
    assert_equal(t.shape, (3, 2))
    assert_raises(ValueError, stats.ttest_rel, rvs1, rvs2, alternative='error')
    t, p = stats.ttest_rel(rvs1, rvs2, axis=0, alternative='less')
    assert_allclose(p, 1 - pr / 2)
    assert_allclose(t, tr)
    t, p = stats.ttest_rel(rvs1, rvs2, axis=0, alternative='greater')
    assert_allclose(p, pr / 2)
    assert_allclose(t, tr)
    rng = np.random.RandomState(12345678)
    x = stats.norm.rvs(loc=5, scale=10, size=501, random_state=rng)
    x[500] = np.nan
    y = stats.norm.rvs(loc=5, scale=10, size=501, random_state=rng) + stats.norm.rvs(scale=0.2, size=501, random_state=rng)
    y[500] = np.nan
    with np.errstate(invalid='ignore'):
        assert_array_equal(stats.ttest_rel(x, x), (np.nan, np.nan))
    assert_array_almost_equal(stats.ttest_rel(x, y, nan_policy='omit'), (0.25299925303978066, 0.8003729814201519))
    assert_raises(ValueError, stats.ttest_rel, x, y, nan_policy='raise')
    assert_raises(ValueError, stats.ttest_rel, x, y, nan_policy='foobar')
    with pytest.warns(RuntimeWarning, match='Precision loss occurred'):
        t, p = stats.ttest_rel([0, 0, 0], [1, 1, 1])
    assert_equal((np.abs(t), p), (np.inf, 0))
    with np.errstate(invalid='ignore'):
        assert_equal(stats.ttest_rel([0, 0, 0], [0, 0, 0]), (np.nan, np.nan))
        anan = np.array([[1, np.nan], [-1, 1]])
        assert_equal(stats.ttest_rel(anan, np.zeros((2, 2))), ([0, np.nan], [1, np.nan]))
    x = np.arange(24)
    assert_raises(ValueError, stats.ttest_rel, x.reshape((8, 3)), x.reshape((2, 3, 4)))

    def convert(t, p, alt):
        if t < 0 and alt == 'less' or (t > 0 and alt == 'greater'):
            return p / 2
        return 1 - p / 2
    converter = np.vectorize(convert)
    rvs1_2D[:, 20:30] = np.nan
    rvs2_2D[:, 15:25] = np.nan
    tr, pr = stats.ttest_rel(rvs1_2D, rvs2_2D, 0, nan_policy='omit')
    t, p = stats.ttest_rel(rvs1_2D, rvs2_2D, 0, nan_policy='omit', alternative='less')
    assert_allclose(t, tr, rtol=1e-14)
    with np.errstate(invalid='ignore'):
        assert_allclose(p, converter(tr, pr, 'less'), rtol=1e-14)
    t, p = stats.ttest_rel(rvs1_2D, rvs2_2D, 0, nan_policy='omit', alternative='greater')
    assert_allclose(t, tr, rtol=1e-14)
    with np.errstate(invalid='ignore'):
        assert_allclose(p, converter(tr, pr, 'greater'), rtol=1e-14)