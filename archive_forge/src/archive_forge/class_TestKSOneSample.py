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
class TestKSOneSample:
    """
    Tests kstest and ks_samp 1-samples with K-S various sizes, alternatives, modes.
    """

    def _testOne(self, x, alternative, expected_statistic, expected_prob, mode='auto', decimal=14):
        result = stats.ks_1samp(x, stats.norm.cdf, alternative=alternative, mode=mode)
        expected = np.array([expected_statistic, expected_prob])
        assert_array_almost_equal(np.array(result), expected, decimal=decimal)

    def test_namedtuple_attributes(self):
        x = np.linspace(-1, 1, 9)
        attributes = ('statistic', 'pvalue')
        res = stats.ks_1samp(x, stats.norm.cdf)
        check_named_results(res, attributes)

    def test_agree_with_r(self):
        x = np.linspace(-1, 1, 9)
        self._testOne(x, 'two-sided', 0.15865525393145705, 0.9516406920151839)
        x = np.linspace(-15, 15, 9)
        self._testOne(x, 'two-sided', 0.4443560271592436, 0.038850140086788665)
        x = [-1.23, 0.06, -0.6, 0.17, 0.66, -0.17, -0.08, 0.27, -0.98, -0.99]
        self._testOne(x, 'two-sided', 0.293580126801961, 0.293408463684361)
        self._testOne(x, 'greater', 0.293580126801961, 0.146988835042376, mode='exact')
        self._testOne(x, 'less', 0.109348552425692, 0.732768892470675, mode='exact')

    def test_known_examples(self):
        x = stats.norm.rvs(loc=0.2, size=100, random_state=987654321)
        self._testOne(x, 'two-sided', 0.12464329735846891, 0.08944488871182077, mode='asymp')
        self._testOne(x, 'less', 0.12464329735846891, 0.04098916407764175)
        self._testOne(x, 'greater', 0.007211523321631099, 0.9853115859039623)

    def test_ks1samp_allpaths(self):
        assert_(np.isnan(kolmogn(np.nan, 1, True)))
        with assert_raises(ValueError, match='n is not integral: 1.5'):
            kolmogn(1.5, 1, True)
        assert_(np.isnan(kolmogn(-1, 1, True)))
        dataset = np.asarray([(101, 1, True, 1.0), (101, 1.1, True, 1.0), (101, 0, True, 0.0), (101, -0.1, True, 0.0), (32, 1.0 / 64, True, 0.0), (32, 1.0 / 64, False, 1.0), (32, 0.5, True, 0.9999999363163307), (32, 0.5, False, 6.368366937916623e-08), (32, 1.0 / 8, True, 0.34624229979775223), (32, 1.0 / 4, True, 0.9699508336558085), (1600, 0.49, False, 0.0), (1600, 1 / 16.0, False, 7.0837876229702195e-06), (1600, 14 / 1600, False, 0.99962357317602), (1600, 1 / 32, False, 0.08603386296651416)])
        FuncData(kolmogn, dataset, (0, 1, 2), 3).check(dtypes=[int, float, bool])

    @pytest.mark.parametrize('ksfunc', [stats.kstest, stats.ks_1samp])
    @pytest.mark.parametrize('alternative, x6val, ref_location, ref_sign', [('greater', 6, 6, +1), ('less', 7, 7, -1), ('two-sided', 6, 6, +1), ('two-sided', 7, 7, -1)])
    def test_location_sign(self, ksfunc, alternative, x6val, ref_location, ref_sign):
        x = np.arange(10) + 0.5
        x[6] = x6val
        cdf = stats.uniform(scale=10).cdf
        res = ksfunc(x, cdf, alternative=alternative)
        assert_allclose(res.statistic, 0.1, rtol=1e-15)
        assert res.statistic_location == ref_location
        assert res.statistic_sign == ref_sign