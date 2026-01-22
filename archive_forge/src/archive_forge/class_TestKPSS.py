from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import MONTH_END, YEAR_END, assert_index_equal
from statsmodels.compat.platform import PLATFORM_WIN
from statsmodels.compat.python import lrange
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas import DataFrame, Series, date_range
import pytest
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.datasets import macrodata, modechoice, nile, randhie, sunspots
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import array_like, bool_like
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (
class TestKPSS:
    """
    R-code
    ------
    library(tseries)
    kpss.stat(x, "Level")
    kpss.stat(x, "Trend")

    In this context, x is the vector containing the
    macrodata['realgdp'] series.
    """

    def setup_method(self):
        self.data = macrodata.load_pandas()
        self.x = self.data.data['realgdp'].values

    def test_fail_nonvector_input(self, reset_randomstate):
        with pytest.warns(InterpolationWarning):
            kpss(self.x, nlags='legacy')
        x = np.random.rand(20, 2)
        assert_raises(ValueError, kpss, x)

    def test_fail_unclear_hypothesis(self):
        with pytest.warns(InterpolationWarning):
            kpss(self.x, 'c', nlags='legacy')
        with pytest.warns(InterpolationWarning):
            kpss(self.x, 'C', nlags='legacy')
        with pytest.warns(InterpolationWarning):
            kpss(self.x, 'ct', nlags='legacy')
        with pytest.warns(InterpolationWarning):
            kpss(self.x, 'CT', nlags='legacy')
        assert_raises(ValueError, kpss, self.x, 'unclear hypothesis', nlags='legacy')

    def test_teststat(self):
        with pytest.warns(InterpolationWarning):
            kpss_stat, _, _, _ = kpss(self.x, 'c', 3)
        assert_almost_equal(kpss_stat, 5.0169, DECIMAL_3)
        with pytest.warns(InterpolationWarning):
            kpss_stat, _, _, _ = kpss(self.x, 'ct', 3)
        assert_almost_equal(kpss_stat, 1.1828, DECIMAL_3)

    def test_pval(self):
        with pytest.warns(InterpolationWarning):
            _, pval, _, _ = kpss(self.x, 'c', 3)
        assert_equal(pval, 0.01)
        with pytest.warns(InterpolationWarning):
            _, pval, _, _ = kpss(self.x, 'ct', 3)
        assert_equal(pval, 0.01)

    def test_store(self):
        with pytest.warns(InterpolationWarning):
            _, _, _, store = kpss(self.x, 'c', 3, True)
        assert_equal(store.nobs, len(self.x))
        assert_equal(store.lags, 3)

    def test_lags(self):
        with pytest.warns(InterpolationWarning):
            res = kpss(self.x, 'c', nlags='auto')
        assert_equal(res[2], 9)
        res = kpss(sunspots.load().data['SUNACTIVITY'], 'c', nlags='auto')
        assert_equal(res[2], 7)
        with pytest.warns(InterpolationWarning):
            res = kpss(nile.load().data['volume'], 'c', nlags='auto')
        assert_equal(res[2], 5)
        with pytest.warns(InterpolationWarning):
            res = kpss(randhie.load().data['lncoins'], 'ct', nlags='auto')
        assert_equal(res[2], 75)
        with pytest.warns(InterpolationWarning):
            res = kpss(modechoice.load().data['invt'], 'ct', nlags='auto')
        assert_equal(res[2], 18)

    def test_kpss_fails_on_nobs_check(self):
        nobs = len(self.x)
        msg = 'lags \\({}\\) must be < number of observations \\({}\\)'.format(nobs, nobs)
        with pytest.raises(ValueError, match=msg):
            kpss(self.x, 'c', nlags=nobs)

    def test_kpss_autolags_does_not_assign_lags_equal_to_nobs(self):
        base = np.array([0, 0, 0, 0, 0, 1, 1.0])
        data_which_breaks_autolag = np.r_[np.tile(base, 297 // 7), [0, 0, 0]]
        kpss(data_which_breaks_autolag, nlags='auto')

    def test_legacy_lags(self):
        with pytest.warns(InterpolationWarning):
            res = kpss(self.x, 'c', nlags='legacy')
        assert_equal(res[2], 15)

    def test_unknown_lags(self):
        with pytest.raises(ValueError):
            kpss(self.x, 'c', nlags='unknown')

    def test_none(self):
        with pytest.warns(FutureWarning):
            kpss(self.x, nlags=None)