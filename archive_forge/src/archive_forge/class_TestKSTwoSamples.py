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
class TestKSTwoSamples:
    """Tests 2-samples with K-S various sizes, alternatives, modes."""

    def _testOne(self, x1, x2, alternative, expected_statistic, expected_prob, mode='auto'):
        result = stats.ks_2samp(x1, x2, alternative, mode=mode)
        expected = np.array([expected_statistic, expected_prob])
        assert_array_almost_equal(np.array(result), expected)

    def testSmall(self):
        self._testOne([0], [1], 'two-sided', 1.0 / 1, 1.0)
        self._testOne([0], [1], 'greater', 1.0 / 1, 0.5)
        self._testOne([0], [1], 'less', 0.0 / 1, 1.0)
        self._testOne([1], [0], 'two-sided', 1.0 / 1, 1.0)
        self._testOne([1], [0], 'greater', 0.0 / 1, 1.0)
        self._testOne([1], [0], 'less', 1.0 / 1, 0.5)

    def testTwoVsThree(self):
        data1 = np.array([1.0, 2.0])
        data1p = data1 + 0.01
        data1m = data1 - 0.01
        data2 = np.array([1.0, 2.0, 3.0])
        self._testOne(data1p, data2, 'two-sided', 1.0 / 3, 1.0)
        self._testOne(data1p, data2, 'greater', 1.0 / 3, 0.7)
        self._testOne(data1p, data2, 'less', 1.0 / 3, 0.7)
        self._testOne(data1m, data2, 'two-sided', 2.0 / 3, 0.6)
        self._testOne(data1m, data2, 'greater', 2.0 / 3, 0.3)
        self._testOne(data1m, data2, 'less', 0, 1.0)

    def testTwoVsFour(self):
        data1 = np.array([1.0, 2.0])
        data1p = data1 + 0.01
        data1m = data1 - 0.01
        data2 = np.array([1.0, 2.0, 3.0, 4.0])
        self._testOne(data1p, data2, 'two-sided', 2.0 / 4, 14.0 / 15)
        self._testOne(data1p, data2, 'greater', 2.0 / 4, 8.0 / 15)
        self._testOne(data1p, data2, 'less', 1.0 / 4, 12.0 / 15)
        self._testOne(data1m, data2, 'two-sided', 3.0 / 4, 6.0 / 15)
        self._testOne(data1m, data2, 'greater', 3.0 / 4, 3.0 / 15)
        self._testOne(data1m, data2, 'less', 0, 1.0)

    def test100_100(self):
        x100 = np.linspace(1, 100, 100)
        x100_2_p1 = x100 + 2 + 0.1
        x100_2_m1 = x100 + 2 - 0.1
        self._testOne(x100, x100_2_p1, 'two-sided', 3.0 / 100, 0.9999999999962055)
        self._testOne(x100, x100_2_p1, 'greater', 3.0 / 100, 0.9143290114276248)
        self._testOne(x100, x100_2_p1, 'less', 0, 1.0)
        self._testOne(x100, x100_2_m1, 'two-sided', 2.0 / 100, 1.0)
        self._testOne(x100, x100_2_m1, 'greater', 2.0 / 100, 0.960978450786184)
        self._testOne(x100, x100_2_m1, 'less', 0, 1.0)

    def test100_110(self):
        x100 = np.linspace(1, 100, 100)
        x110 = np.linspace(1, 100, 110)
        x110_20_p1 = x110 + 20 + 0.1
        x110_20_m1 = x110 + 20 - 0.1
        self._testOne(x100, x110_20_p1, 'two-sided', 232.0 / 1100, 0.015739183865607353)
        self._testOne(x100, x110_20_p1, 'greater', 232.0 / 1100, 0.007869594319053203)
        self._testOne(x100, x110_20_p1, 'less', 0, 1)
        self._testOne(x100, x110_20_m1, 'two-sided', 229.0 / 1100, 0.017803803861026313)
        self._testOne(x100, x110_20_m1, 'greater', 229.0 / 1100, 0.008901905958245056)
        self._testOne(x100, x110_20_m1, 'less', 0.0, 1.0)

    def testRepeatedValues(self):
        x2233 = np.array([2] * 3 + [3] * 4 + [5] * 5 + [6] * 4, dtype=int)
        x3344 = x2233 + 1
        x2356 = np.array([2] * 3 + [3] * 4 + [5] * 10 + [6] * 4, dtype=int)
        x3467 = np.array([3] * 10 + [4] * 2 + [6] * 10 + [7] * 4, dtype=int)
        self._testOne(x2233, x3344, 'two-sided', 5.0 / 16, 0.4262934613454952)
        self._testOne(x2233, x3344, 'greater', 5.0 / 16, 0.21465428276573786)
        self._testOne(x2233, x3344, 'less', 0.0 / 16, 1.0)
        self._testOne(x2356, x3467, 'two-sided', 190.0 / 21 / 26, 0.0919245790168125)
        self._testOne(x2356, x3467, 'greater', 190.0 / 21 / 26, 0.0459633806858544)
        self._testOne(x2356, x3467, 'less', 70.0 / 21 / 26, 0.6121593130022775)

    def testEqualSizes(self):
        data2 = np.array([1.0, 2.0, 3.0])
        self._testOne(data2, data2 + 1, 'two-sided', 1.0 / 3, 1.0)
        self._testOne(data2, data2 + 1, 'greater', 1.0 / 3, 0.75)
        self._testOne(data2, data2 + 1, 'less', 0.0 / 3, 1.0)
        self._testOne(data2, data2 + 0.5, 'two-sided', 1.0 / 3, 1.0)
        self._testOne(data2, data2 + 0.5, 'greater', 1.0 / 3, 0.75)
        self._testOne(data2, data2 + 0.5, 'less', 0.0 / 3, 1.0)
        self._testOne(data2, data2 - 0.5, 'two-sided', 1.0 / 3, 1.0)
        self._testOne(data2, data2 - 0.5, 'greater', 0.0 / 3, 1.0)
        self._testOne(data2, data2 - 0.5, 'less', 1.0 / 3, 0.75)

    @pytest.mark.slow
    def testMiddlingBoth(self):
        n1, n2 = (500, 600)
        delta = 1.0 / n1 / n2 / 2 / 2
        x = np.linspace(1, 200, n1) - delta
        y = np.linspace(2, 200, n2)
        self._testOne(x, y, 'two-sided', 2000.0 / n1 / n2, 1.0, mode='auto')
        self._testOne(x, y, 'two-sided', 2000.0 / n1 / n2, 1.0, mode='asymp')
        self._testOne(x, y, 'greater', 2000.0 / n1 / n2, 0.9697596024683929, mode='asymp')
        self._testOne(x, y, 'less', 500.0 / n1 / n2, 0.9968735843165021, mode='asymp')
        with suppress_warnings() as sup:
            message = 'ks_2samp: Exact calculation unsuccessful.'
            sup.filter(RuntimeWarning, message)
            self._testOne(x, y, 'greater', 2000.0 / n1 / n2, 0.9697596024683929, mode='exact')
            self._testOne(x, y, 'less', 500.0 / n1 / n2, 0.9968735843165021, mode='exact')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self._testOne(x, y, 'less', 500.0 / n1 / n2, 0.9968735843165021, mode='exact')
            _check_warnings(w, RuntimeWarning, 1)

    @pytest.mark.slow
    def testMediumBoth(self):
        n1, n2 = (1000, 1100)
        delta = 1.0 / n1 / n2 / 2 / 2
        x = np.linspace(1, 200, n1) - delta
        y = np.linspace(2, 200, n2)
        self._testOne(x, y, 'two-sided', 6600.0 / n1 / n2, 1.0, mode='asymp')
        self._testOne(x, y, 'two-sided', 6600.0 / n1 / n2, 1.0, mode='auto')
        self._testOne(x, y, 'greater', 6600.0 / n1 / n2, 0.9573185808092622, mode='asymp')
        self._testOne(x, y, 'less', 1000.0 / n1 / n2, 0.9982410869433984, mode='asymp')
        with suppress_warnings() as sup:
            message = 'ks_2samp: Exact calculation unsuccessful.'
            sup.filter(RuntimeWarning, message)
            self._testOne(x, y, 'greater', 6600.0 / n1 / n2, 0.9573185808092622, mode='exact')
            self._testOne(x, y, 'less', 1000.0 / n1 / n2, 0.9982410869433984, mode='exact')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self._testOne(x, y, 'less', 1000.0 / n1 / n2, 0.9982410869433984, mode='exact')
            _check_warnings(w, RuntimeWarning, 1)

    def testLarge(self):
        n1, n2 = (10000, 110)
        lcm = n1 * 11.0
        delta = 1.0 / n1 / n2 / 2 / 2
        x = np.linspace(1, 200, n1) - delta
        y = np.linspace(2, 100, n2)
        self._testOne(x, y, 'two-sided', 55275.0 / lcm, 4.218847493575595e-15)
        self._testOne(x, y, 'greater', 561.0 / lcm, 0.9911545458204759)
        self._testOne(x, y, 'less', 55275.0 / lcm, 3.1317328311518713e-26)

    def test_gh11184(self):
        np.random.seed(123456)
        x = np.random.normal(size=3000)
        y = np.random.normal(size=3001) * 1.5
        self._testOne(x, y, 'two-sided', 0.11292880151060758, 2.7755575615628914e-15, mode='asymp')
        self._testOne(x, y, 'two-sided', 0.11292880151060758, 2.7755575615628914e-15, mode='exact')

    @pytest.mark.xslow
    def test_gh11184_bigger(self):
        np.random.seed(123456)
        x = np.random.normal(size=10000)
        y = np.random.normal(size=10001) * 1.5
        self._testOne(x, y, 'two-sided', 0.10597913208679133, 3.3149311398483503e-49, mode='asymp')
        self._testOne(x, y, 'two-sided', 0.10597913208679133, 2.7755575615628914e-15, mode='exact')
        self._testOne(x, y, 'greater', 0.10597913208679133, 2.7947433906389253e-41, mode='asymp')
        self._testOne(x, y, 'less', 0.09658002199780022, 2.7947433906389253e-41, mode='asymp')

    @pytest.mark.xslow
    def test_gh12999(self):
        np.random.seed(123456)
        for x in range(1000, 12000, 1000):
            vals1 = np.random.normal(size=x)
            vals2 = np.random.normal(size=x + 10, loc=0.5)
            exact = stats.ks_2samp(vals1, vals2, mode='exact').pvalue
            asymp = stats.ks_2samp(vals1, vals2, mode='asymp').pvalue
            assert_array_less(exact, 3 * asymp)
            assert_array_less(asymp, 3 * exact)

    @pytest.mark.slow
    def testLargeBoth(self):
        n1, n2 = (10000, 11000)
        lcm = n1 * 11.0
        delta = 1.0 / n1 / n2 / 2 / 2
        x = np.linspace(1, 200, n1) - delta
        y = np.linspace(2, 200, n2)
        self._testOne(x, y, 'two-sided', 563.0 / lcm, 0.9990660108966576, mode='asymp')
        self._testOne(x, y, 'two-sided', 563.0 / lcm, 0.9990456491488628, mode='exact')
        self._testOne(x, y, 'two-sided', 563.0 / lcm, 0.9990660108966576, mode='auto')
        self._testOne(x, y, 'greater', 563.0 / lcm, 0.7561851877420673)
        self._testOne(x, y, 'less', 10.0 / lcm, 0.9998239693191724)
        with suppress_warnings() as sup:
            message = 'ks_2samp: Exact calculation unsuccessful.'
            sup.filter(RuntimeWarning, message)
            self._testOne(x, y, 'greater', 563.0 / lcm, 0.7561851877420673, mode='exact')
            self._testOne(x, y, 'less', 10.0 / lcm, 0.9998239693191724, mode='exact')

    def testNamedAttributes(self):
        attributes = ('statistic', 'pvalue')
        res = stats.ks_2samp([1, 2], [3])
        check_named_results(res, attributes)

    @pytest.mark.slow
    def test_some_code_paths(self):
        from scipy.stats._stats_py import _count_paths_outside_method, _compute_outer_prob_inside_method
        _compute_outer_prob_inside_method(1, 1, 1, 1)
        _count_paths_outside_method(1000, 1, 1, 1001)
        with np.errstate(invalid='raise'):
            assert_raises(FloatingPointError, _count_paths_outside_method, 1100, 1099, 1, 1)
            assert_raises(FloatingPointError, _count_paths_outside_method, 2000, 1000, 1, 1)

    def test_argument_checking(self):
        assert_raises(ValueError, stats.ks_2samp, [], [1])
        assert_raises(ValueError, stats.ks_2samp, [1], [])
        assert_raises(ValueError, stats.ks_2samp, [], [])

    @pytest.mark.slow
    def test_gh12218(self):
        """Ensure gh-12218 is fixed."""
        np.random.seed(12345678)
        n1 = 2097152
        rvs1 = stats.uniform.rvs(size=n1, loc=0.0, scale=1)
        rvs2 = rvs1 + 1
        stats.ks_2samp(rvs1, rvs2, alternative='greater', mode='asymp')
        stats.ks_2samp(rvs1, rvs2, alternative='less', mode='asymp')
        stats.ks_2samp(rvs1, rvs2, alternative='two-sided', mode='asymp')

    def test_warnings_gh_14019(self):
        rng = np.random.default_rng(abs(hash('test_warnings_gh_14019')))
        data1 = rng.random(size=881) + 0.5
        data2 = rng.random(size=369)
        message = 'ks_2samp: Exact calculation unsuccessful'
        with pytest.warns(RuntimeWarning, match=message):
            res = stats.ks_2samp(data1, data2, alternative='less')
            assert_allclose(res.pvalue, 0, atol=1e-14)

    @pytest.mark.parametrize('ksfunc', [stats.kstest, stats.ks_2samp])
    @pytest.mark.parametrize('alternative, x6val, ref_location, ref_sign', [('greater', 5.9, 5.9, +1), ('less', 6.1, 6.0, -1), ('two-sided', 5.9, 5.9, +1), ('two-sided', 6.1, 6.0, -1)])
    def test_location_sign(self, ksfunc, alternative, x6val, ref_location, ref_sign):
        x = np.arange(10, dtype=np.float64)
        y = x.copy()
        x[6] = x6val
        res = stats.ks_2samp(x, y, alternative=alternative)
        assert res.statistic == 0.1
        assert res.statistic_location == ref_location
        assert res.statistic_sign == ref_sign