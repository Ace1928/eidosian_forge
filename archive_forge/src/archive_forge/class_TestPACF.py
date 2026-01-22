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
class TestPACF(CheckCorrGram):

    @classmethod
    def setup_class(cls):
        cls.pacfols = cls.results['PACOLS']
        cls.pacfyw = cls.results['PACYW']

    def test_ols(self):
        pacfols, confint = pacf(self.x, nlags=40, alpha=0.05, method='ols')
        assert_almost_equal(pacfols[1:], self.pacfols, DECIMAL_6)
        centered = confint - confint.mean(1)[:, None]
        res = [[-0.1375625, 0.1375625]] * 40
        assert_almost_equal(centered[1:41], res, DECIMAL_6)
        assert_equal(centered[0], [0.0, 0.0])
        assert_equal(confint[0], [1, 1])
        assert_equal(pacfols[0], 1)

    def test_ols_inefficient(self):
        lag_len = 5
        pacfols = pacf_ols(self.x, nlags=lag_len, efficient=False)
        x = self.x.copy()
        x -= x.mean()
        n = x.shape[0]
        lags = np.zeros((n - 5, 5))
        lead = x[5:]
        direct = np.empty(lag_len + 1)
        direct[0] = 1.0
        for i in range(lag_len):
            lags[:, i] = x[5 - (i + 1):-(i + 1)]
            direct[i + 1] = lstsq(lags[:, :i + 1], lead, rcond=None)[0][-1]
        assert_allclose(pacfols, direct, atol=1e-08)

    def test_yw(self):
        pacfyw = pacf_yw(self.x, nlags=40, method='mle')
        assert_almost_equal(pacfyw[1:], self.pacfyw, DECIMAL_8)

    def test_yw_singular(self):
        with pytest.warns(ValueWarning):
            pacf(np.ones(30), nlags=6)

    def test_ld(self):
        pacfyw = pacf_yw(self.x, nlags=40, method='mle')
        pacfld = pacf(self.x, nlags=40, method='ldb')
        assert_almost_equal(pacfyw, pacfld, DECIMAL_8)
        pacfyw = pacf(self.x, nlags=40, method='yw')
        pacfld = pacf(self.x, nlags=40, method='lda')
        assert_almost_equal(pacfyw, pacfld, DECIMAL_8)

    def test_burg(self):
        pacfburg_, _ = pacf_burg(self.x, nlags=40)
        pacfburg = pacf(self.x, nlags=40, method='burg')
        assert_almost_equal(pacfburg_, pacfburg, DECIMAL_8)