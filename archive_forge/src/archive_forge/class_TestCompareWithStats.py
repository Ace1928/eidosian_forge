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
class TestCompareWithStats:
    """
    Class to compare mstats results with stats results.

    It is in general assumed that scipy.stats is at a more mature stage than
    stats.mstats.  If a routine in mstats results in similar results like in
    scipy.stats, this is considered also as a proper validation of scipy.mstats
    routine.

    Different sample sizes are used for testing, as some problems between stats
    and mstats are dependent on sample size.

    Author: Alexander Loew

    NOTE that some tests fail. This might be caused by
    a) actual differences or bugs between stats and mstats
    b) numerical inaccuracies
    c) different definitions of routine interfaces

    These failures need to be checked. Current workaround is to have disabled these
    tests, but issuing reports on scipy-dev

    """

    def get_n(self):
        """ Returns list of sample sizes to be used for comparison. """
        return [1000, 100, 10, 5]

    def generate_xy_sample(self, n):
        np.random.seed(1234567)
        x = np.random.randn(n)
        y = x + np.random.randn(n)
        xm = np.full(len(x) + 5, 1e+16)
        ym = np.full(len(y) + 5, 1e+16)
        xm[0:len(x)] = x
        ym[0:len(y)] = y
        mask = xm > 9000000000000000.0
        xm = np.ma.array(xm, mask=mask)
        ym = np.ma.array(ym, mask=mask)
        return (x, y, xm, ym)

    def generate_xy_sample2D(self, n, nx):
        x = np.full((n, nx), np.nan)
        y = np.full((n, nx), np.nan)
        xm = np.full((n + 5, nx), np.nan)
        ym = np.full((n + 5, nx), np.nan)
        for i in range(nx):
            x[:, i], y[:, i], dx, dy = self.generate_xy_sample(n)
        xm[0:n, :] = x[0:n]
        ym[0:n, :] = y[0:n]
        xm = np.ma.array(xm, mask=np.isnan(xm))
        ym = np.ma.array(ym, mask=np.isnan(ym))
        return (x, y, xm, ym)

    def test_linregress(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            result1 = stats.linregress(x, y)
            result2 = stats.mstats.linregress(xm, ym)
            assert_allclose(np.asarray(result1), np.asarray(result2))

    def test_pearsonr(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r, p = stats.pearsonr(x, y)
            rm, pm = stats.mstats.pearsonr(xm, ym)
            assert_almost_equal(r, rm, decimal=14)
            assert_almost_equal(p, pm, decimal=14)

    def test_spearmanr(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r, p = stats.spearmanr(x, y)
            rm, pm = stats.mstats.spearmanr(xm, ym)
            assert_almost_equal(r, rm, 14)
            assert_almost_equal(p, pm, 14)

    def test_spearmanr_backcompat_useties(self):
        x = np.arange(6)
        assert_raises(ValueError, mstats.spearmanr, x, x, False)

    def test_gmean(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.gmean(abs(x))
            rm = stats.mstats.gmean(abs(xm))
            assert_allclose(r, rm, rtol=1e-13)
            r = stats.gmean(abs(y))
            rm = stats.mstats.gmean(abs(ym))
            assert_allclose(r, rm, rtol=1e-13)

    def test_hmean(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.hmean(abs(x))
            rm = stats.mstats.hmean(abs(xm))
            assert_almost_equal(r, rm, 10)
            r = stats.hmean(abs(y))
            rm = stats.mstats.hmean(abs(ym))
            assert_almost_equal(r, rm, 10)

    def test_skew(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.skew(x)
            rm = stats.mstats.skew(xm)
            assert_almost_equal(r, rm, 10)
            r = stats.skew(y)
            rm = stats.mstats.skew(ym)
            assert_almost_equal(r, rm, 10)

    def test_moment(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.moment(x)
            rm = stats.mstats.moment(xm)
            assert_almost_equal(r, rm, 10)
            r = stats.moment(y)
            rm = stats.mstats.moment(ym)
            assert_almost_equal(r, rm, 10)

    def test_zscore(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            zx = (x - x.mean()) / x.std()
            zy = (y - y.mean()) / y.std()
            assert_allclose(stats.zscore(x), zx, rtol=1e-10)
            assert_allclose(stats.zscore(y), zy, rtol=1e-10)
            assert_allclose(stats.zscore(x), stats.mstats.zscore(xm[0:len(x)]), rtol=1e-10)
            assert_allclose(stats.zscore(y), stats.mstats.zscore(ym[0:len(y)]), rtol=1e-10)

    def test_kurtosis(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.kurtosis(x)
            rm = stats.mstats.kurtosis(xm)
            assert_almost_equal(r, rm, 10)
            r = stats.kurtosis(y)
            rm = stats.mstats.kurtosis(ym)
            assert_almost_equal(r, rm, 10)

    def test_sem(self):
        a = np.arange(20).reshape(5, 4)
        am = np.ma.array(a)
        r = stats.sem(a, ddof=1)
        rm = stats.mstats.sem(am, ddof=1)
        assert_allclose(r, 2.82842712, atol=1e-05)
        assert_allclose(rm, 2.82842712, atol=1e-05)
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_almost_equal(stats.mstats.sem(xm, axis=None, ddof=0), stats.sem(x, axis=None, ddof=0), decimal=13)
            assert_almost_equal(stats.mstats.sem(ym, axis=None, ddof=0), stats.sem(y, axis=None, ddof=0), decimal=13)
            assert_almost_equal(stats.mstats.sem(xm, axis=None, ddof=1), stats.sem(x, axis=None, ddof=1), decimal=13)
            assert_almost_equal(stats.mstats.sem(ym, axis=None, ddof=1), stats.sem(y, axis=None, ddof=1), decimal=13)

    def test_describe(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.describe(x, ddof=1)
            rm = stats.mstats.describe(xm, ddof=1)
            for ii in range(6):
                assert_almost_equal(np.asarray(r[ii]), np.asarray(rm[ii]), decimal=12)

    def test_describe_result_attributes(self):
        actual = mstats.describe(np.arange(5))
        attributes = ('nobs', 'minmax', 'mean', 'variance', 'skewness', 'kurtosis')
        check_named_results(actual, attributes, ma=True)

    def test_rankdata(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.rankdata(x)
            rm = stats.mstats.rankdata(x)
            assert_allclose(r, rm)

    def test_tmean(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_almost_equal(stats.tmean(x), stats.mstats.tmean(xm), 14)
            assert_almost_equal(stats.tmean(y), stats.mstats.tmean(ym), 14)

    def test_tmax(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_almost_equal(stats.tmax(x, 2.0), stats.mstats.tmax(xm, 2.0), 10)
            assert_almost_equal(stats.tmax(y, 2.0), stats.mstats.tmax(ym, 2.0), 10)
            assert_almost_equal(stats.tmax(x, upperlimit=3.0), stats.mstats.tmax(xm, upperlimit=3.0), 10)
            assert_almost_equal(stats.tmax(y, upperlimit=3.0), stats.mstats.tmax(ym, upperlimit=3.0), 10)

    def test_tmin(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_equal(stats.tmin(x), stats.mstats.tmin(xm))
            assert_equal(stats.tmin(y), stats.mstats.tmin(ym))
            assert_almost_equal(stats.tmin(x, lowerlimit=-1.0), stats.mstats.tmin(xm, lowerlimit=-1.0), 10)
            assert_almost_equal(stats.tmin(y, lowerlimit=-1.0), stats.mstats.tmin(ym, lowerlimit=-1.0), 10)

    def test_zmap(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            z = stats.zmap(x, y)
            zm = stats.mstats.zmap(xm, ym)
            assert_allclose(z, zm[0:len(z)], atol=1e-10)

    def test_variation(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_almost_equal(stats.variation(x), stats.mstats.variation(xm), decimal=12)
            assert_almost_equal(stats.variation(y), stats.mstats.variation(ym), decimal=12)

    def test_tvar(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_almost_equal(stats.tvar(x), stats.mstats.tvar(xm), decimal=12)
            assert_almost_equal(stats.tvar(y), stats.mstats.tvar(ym), decimal=12)

    def test_trimboth(self):
        a = np.arange(20)
        b = stats.trimboth(a, 0.1)
        bm = stats.mstats.trimboth(a, 0.1)
        assert_allclose(np.sort(b), bm.data[~bm.mask])

    def test_tsem(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_almost_equal(stats.tsem(x), stats.mstats.tsem(xm), decimal=14)
            assert_almost_equal(stats.tsem(y), stats.mstats.tsem(ym), decimal=14)
            assert_almost_equal(stats.tsem(x, limits=(-2.0, 2.0)), stats.mstats.tsem(xm, limits=(-2.0, 2.0)), decimal=14)

    def test_skewtest(self):
        for n in self.get_n():
            if n > 8:
                x, y, xm, ym = self.generate_xy_sample(n)
                r = stats.skewtest(x)
                rm = stats.mstats.skewtest(xm)
                assert_allclose(r, rm)

    def test_skewtest_result_attributes(self):
        x = np.array((-2, -1, 0, 1, 2, 3) * 4) ** 2
        res = mstats.skewtest(x)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_skewtest_2D_notmasked(self):
        x = np.random.random((20, 2)) * 20.0
        r = stats.skewtest(x)
        rm = stats.mstats.skewtest(x)
        assert_allclose(np.asarray(r), np.asarray(rm))

    def test_skewtest_2D_WithMask(self):
        nx = 2
        for n in self.get_n():
            if n > 8:
                x, y, xm, ym = self.generate_xy_sample2D(n, nx)
                r = stats.skewtest(x)
                rm = stats.mstats.skewtest(xm)
                assert_allclose(r[0][0], rm[0][0], rtol=1e-14)
                assert_allclose(r[0][1], rm[0][1], rtol=1e-14)

    def test_normaltest(self):
        with np.errstate(over='raise'), suppress_warnings() as sup:
            sup.filter(UserWarning, 'kurtosistest only valid for n>=20')
            for n in self.get_n():
                if n > 8:
                    x, y, xm, ym = self.generate_xy_sample(n)
                    r = stats.normaltest(x)
                    rm = stats.mstats.normaltest(xm)
                    assert_allclose(np.asarray(r), np.asarray(rm))

    def test_find_repeats(self):
        x = np.asarray([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4]).astype('float')
        tmp = np.asarray([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]).astype('float')
        mask = tmp == 5.0
        xm = np.ma.array(tmp, mask=mask)
        x_orig, xm_orig = (x.copy(), xm.copy())
        r = stats.find_repeats(x)
        rm = stats.mstats.find_repeats(xm)
        assert_equal(r, rm)
        assert_equal(x, x_orig)
        assert_equal(xm, xm_orig)
        _, counts = stats.mstats.find_repeats([])
        assert_equal(counts, np.array(0, dtype=np.intp))

    def test_kendalltau(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.kendalltau(x, y)
            rm = stats.mstats.kendalltau(xm, ym)
            assert_almost_equal(r[0], rm[0], decimal=10)
            assert_almost_equal(r[1], rm[1], decimal=7)

    def test_obrientransform(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.obrientransform(x)
            rm = stats.mstats.obrientransform(xm)
            assert_almost_equal(r.T, rm[0:len(x)])

    def test_ks_1samp(self):
        """Checks that mstats.ks_1samp and stats.ks_1samp agree on masked arrays."""
        for mode in ['auto', 'exact', 'asymp']:
            with suppress_warnings():
                for alternative in ['less', 'greater', 'two-sided']:
                    for n in self.get_n():
                        x, y, xm, ym = self.generate_xy_sample(n)
                        res1 = stats.ks_1samp(x, stats.norm.cdf, alternative=alternative, mode=mode)
                        res2 = stats.mstats.ks_1samp(xm, stats.norm.cdf, alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res2))
                        res3 = stats.ks_1samp(xm, stats.norm.cdf, alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res3))

    def test_kstest_1samp(self):
        """
        Checks that 1-sample mstats.kstest and stats.kstest agree on masked arrays.
        """
        for mode in ['auto', 'exact', 'asymp']:
            with suppress_warnings():
                for alternative in ['less', 'greater', 'two-sided']:
                    for n in self.get_n():
                        x, y, xm, ym = self.generate_xy_sample(n)
                        res1 = stats.kstest(x, 'norm', alternative=alternative, mode=mode)
                        res2 = stats.mstats.kstest(xm, 'norm', alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res2))
                        res3 = stats.kstest(xm, 'norm', alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res3))

    def test_ks_2samp(self):
        """Checks that mstats.ks_2samp and stats.ks_2samp agree on masked arrays.
        gh-8431"""
        for mode in ['auto', 'exact', 'asymp']:
            with suppress_warnings() as sup:
                if mode in ['auto', 'exact']:
                    message = 'ks_2samp: Exact calculation unsuccessful.'
                    sup.filter(RuntimeWarning, message)
                for alternative in ['less', 'greater', 'two-sided']:
                    for n in self.get_n():
                        x, y, xm, ym = self.generate_xy_sample(n)
                        res1 = stats.ks_2samp(x, y, alternative=alternative, mode=mode)
                        res2 = stats.mstats.ks_2samp(xm, ym, alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res2))
                        res3 = stats.ks_2samp(xm, y, alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res3))

    def test_kstest_2samp(self):
        """
        Checks that 2-sample mstats.kstest and stats.kstest agree on masked arrays.
        """
        for mode in ['auto', 'exact', 'asymp']:
            with suppress_warnings() as sup:
                if mode in ['auto', 'exact']:
                    message = 'ks_2samp: Exact calculation unsuccessful.'
                    sup.filter(RuntimeWarning, message)
                for alternative in ['less', 'greater', 'two-sided']:
                    for n in self.get_n():
                        x, y, xm, ym = self.generate_xy_sample(n)
                        res1 = stats.kstest(x, y, alternative=alternative, mode=mode)
                        res2 = stats.mstats.kstest(xm, ym, alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res2))
                        res3 = stats.kstest(xm, y, alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res3))