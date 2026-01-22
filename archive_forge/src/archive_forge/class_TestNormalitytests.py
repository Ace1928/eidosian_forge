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
class TestNormalitytests:

    def test_vs_nonmasked(self):
        x = np.array((-2, -1, 0, 1, 2, 3) * 4) ** 2
        assert_array_almost_equal(mstats.normaltest(x), stats.normaltest(x))
        assert_array_almost_equal(mstats.skewtest(x), stats.skewtest(x))
        assert_array_almost_equal(mstats.kurtosistest(x), stats.kurtosistest(x))
        funcs = [stats.normaltest, stats.skewtest, stats.kurtosistest]
        mfuncs = [mstats.normaltest, mstats.skewtest, mstats.kurtosistest]
        x = [1, 2, 3, 4]
        for func, mfunc in zip(funcs, mfuncs):
            assert_raises(ValueError, func, x)
            assert_raises(ValueError, mfunc, x)

    def test_axis_None(self):
        x = np.array((-2, -1, 0, 1, 2, 3) * 4) ** 2
        assert_allclose(mstats.normaltest(x, axis=None), mstats.normaltest(x))
        assert_allclose(mstats.skewtest(x, axis=None), mstats.skewtest(x))
        assert_allclose(mstats.kurtosistest(x, axis=None), mstats.kurtosistest(x))

    def test_maskedarray_input(self):
        x = np.array((-2, -1, 0, 1, 2, 3) * 4) ** 2
        xm = np.ma.array(np.r_[np.inf, x, 10], mask=np.r_[True, [False] * x.size, True])
        assert_allclose(mstats.normaltest(xm), stats.normaltest(x))
        assert_allclose(mstats.skewtest(xm), stats.skewtest(x))
        assert_allclose(mstats.kurtosistest(xm), stats.kurtosistest(x))

    def test_nd_input(self):
        x = np.array((-2, -1, 0, 1, 2, 3) * 4) ** 2
        x_2d = np.vstack([x] * 2).T
        for func in [mstats.normaltest, mstats.skewtest, mstats.kurtosistest]:
            res_1d = func(x)
            res_2d = func(x_2d)
            assert_allclose(res_2d[0], [res_1d[0]] * 2)
            assert_allclose(res_2d[1], [res_1d[1]] * 2)

    def test_normaltest_result_attributes(self):
        x = np.array((-2, -1, 0, 1, 2, 3) * 4) ** 2
        res = mstats.normaltest(x)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_kurtosistest_result_attributes(self):
        x = np.array((-2, -1, 0, 1, 2, 3) * 4) ** 2
        res = mstats.kurtosistest(x)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_regression_9033(self):
        counts = [128, 0, 58, 7, 0, 41, 16, 0, 0, 167]
        x = np.hstack([np.full(c, i) for i, c in enumerate(counts)])
        assert_equal(mstats.kurtosistest(x)[1] < 0.01, True)

    @pytest.mark.parametrize('test', ['skewtest', 'kurtosistest'])
    @pytest.mark.parametrize('alternative', ['less', 'greater'])
    def test_alternative(self, test, alternative):
        x = stats.norm.rvs(loc=10, scale=2.5, size=30, random_state=123)
        stats_test = getattr(stats, test)
        mstats_test = getattr(mstats, test)
        z_ex, p_ex = stats_test(x, alternative=alternative)
        z, p = mstats_test(x, alternative=alternative)
        assert_allclose(z, z_ex, atol=1e-12)
        assert_allclose(p, p_ex, atol=1e-12)
        x[1:5] = np.nan
        x = np.ma.masked_array(x, mask=np.isnan(x))
        z_ex, p_ex = stats_test(x.compressed(), alternative=alternative)
        z, p = mstats_test(x, alternative=alternative)
        assert_allclose(z, z_ex, atol=1e-12)
        assert_allclose(p, p_ex, atol=1e-12)

    def test_bad_alternative(self):
        x = stats.norm.rvs(size=20, random_state=123)
        msg = "alternative must be 'less', 'greater' or 'two-sided'"
        with pytest.raises(ValueError, match=msg):
            mstats.skewtest(x, alternative='error')
        with pytest.raises(ValueError, match=msg):
            mstats.kurtosistest(x, alternative='error')