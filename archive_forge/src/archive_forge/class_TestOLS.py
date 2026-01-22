from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
class TestOLS(CheckRegressionResults):

    @classmethod
    def setup_class(cls):
        from .results.results_regression import Longley
        data = longley.load()
        endog = np.asarray(data.endog)
        exog = np.asarray(data.exog)
        exog = add_constant(exog, prepend=False)
        res1 = OLS(endog, exog).fit()
        res2 = Longley()
        res2.wresid = res1.wresid
        cls.res1 = res1
        cls.res2 = res2
        res_qr = OLS(endog, exog).fit(method='qr')
        model_qr = OLS(endog, exog)
        Q, R = np.linalg.qr(exog)
        model_qr.exog_Q, model_qr.exog_R = (Q, R)
        model_qr.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))
        model_qr.rank = np.linalg.matrix_rank(R)
        res_qr2 = model_qr.fit(method='qr')
        cls.res_qr = res_qr
        cls.res_qr_manual = res_qr2

    def test_eigenvalues(self):
        eigenval_perc_diff = self.res_qr.eigenvals - self.res_qr_manual.eigenvals
        eigenval_perc_diff /= self.res_qr.eigenvals
        zeros = np.zeros_like(eigenval_perc_diff)
        assert_almost_equal(eigenval_perc_diff, zeros, DECIMAL_7)

    def test_HC0_errors(self):
        assert_almost_equal(self.res1.HC0_se[:-1], self.res2.HC0_se[:-1], DECIMAL_4)
        assert_allclose(np.round(self.res1.HC0_se[-1]), self.res2.HC0_se[-1])

    def test_HC1_errors(self):
        assert_almost_equal(self.res1.HC1_se[:-1], self.res2.HC1_se[:-1], DECIMAL_4)
        assert_allclose(self.res1.HC1_se[-1], self.res2.HC1_se[-1], rtol=4e-07)

    def test_HC2_errors(self):
        assert_almost_equal(self.res1.HC2_se[:-1], self.res2.HC2_se[:-1], DECIMAL_4)
        assert_allclose(self.res1.HC2_se[-1], self.res2.HC2_se[-1], rtol=5e-07)

    def test_HC3_errors(self):
        assert_almost_equal(self.res1.HC3_se[:-1], self.res2.HC3_se[:-1], DECIMAL_4)
        assert_allclose(self.res1.HC3_se[-1], self.res2.HC3_se[-1], rtol=1.5e-07)

    def test_qr_params(self):
        assert_almost_equal(self.res1.params, self.res_qr.params, 6)

    def test_qr_normalized_cov_params(self):
        assert_almost_equal(np.ones_like(self.res1.normalized_cov_params), self.res1.normalized_cov_params / self.res_qr.normalized_cov_params, 5)

    def test_missing(self):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        data.endog[[3, 7, 14]] = np.nan
        mod = OLS(data.endog, data.exog, missing='drop')
        assert_equal(mod.endog.shape[0], 13)
        assert_equal(mod.exog.shape[0], 13)

    def test_rsquared_adj_overfit(self):
        with warnings.catch_warnings(record=True):
            x = np.random.randn(5)
            y = np.random.randn(5, 6)
            results = OLS(x, y).fit()
            rsquared_adj = results.rsquared_adj
            assert_equal(rsquared_adj, np.nan)

    def test_qr_alternatives(self):
        assert_allclose(self.res_qr.params, self.res_qr_manual.params, rtol=5e-12)

    def test_norm_resid(self):
        resid = self.res1.wresid
        norm_resid = resid / np.sqrt(np.sum(resid ** 2.0) / self.res1.df_resid)
        model_norm_resid = self.res1.resid_pearson
        assert_almost_equal(model_norm_resid, norm_resid, DECIMAL_7)

    def test_summary_slim(self):
        with warnings.catch_warnings():
            msg = 'kurtosistest only valid for n>=20'
            warnings.filterwarnings('ignore', message=msg, category=UserWarning)
            summ = self.res1.summary(slim=True)
        assert len(summ.tables) == 2
        assert len(str(summ)) < 6700

    def test_norm_resid_zero_variance(self):
        with warnings.catch_warnings(record=True):
            y = self.res1.model.endog
            res = OLS(y, y).fit()
            assert_allclose(res.scale, 0, atol=1e-20)
            assert_allclose(res.wresid, res.resid_pearson, atol=5e-11)