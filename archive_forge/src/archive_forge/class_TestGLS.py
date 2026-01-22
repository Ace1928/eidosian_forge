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
class TestGLS:
    """
    These test results were obtained by replication with R.
    """

    @classmethod
    def setup_class(cls):
        from .results.results_regression import LongleyGls
        data = longley.load()
        exog = add_constant(np.column_stack((data.exog.iloc[:, 1], data.exog.iloc[:, 4])), prepend=False)
        tmp_results = OLS(data.endog, exog).fit()
        rho = np.corrcoef(tmp_results.resid[1:], tmp_results.resid[:-1])[0][1]
        order = toeplitz(np.arange(16))
        sigma = rho ** order
        GLS_results = GLS(data.endog, exog, sigma=sigma).fit()
        cls.res1 = GLS_results
        cls.res2 = LongleyGls()
        cls.sigma = sigma
        cls.exog = exog
        cls.endog = data.endog

    def test_aic(self):
        assert_allclose(self.res1.aic + 2, self.res2.aic, rtol=0.004)

    def test_bic(self):
        assert_allclose(self.res1.bic, self.res2.bic, rtol=0.015)

    def test_loglike(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_0)

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_1)

    def test_resid(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, DECIMAL_4)

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale, DECIMAL_4)

    def test_tvalues(self):
        assert_almost_equal(self.res1.tvalues, self.res2.tvalues, DECIMAL_4)

    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)

    def test_fittedvalues(self):
        assert_almost_equal(self.res1.fittedvalues, self.res2.fittedvalues, DECIMAL_4)

    def test_pvalues(self):
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL_4)

    def test_missing(self):
        endog = self.endog.copy()
        endog[[4, 7, 14]] = np.nan
        mod = GLS(endog, self.exog, sigma=self.sigma, missing='drop')
        assert_equal(mod.endog.shape[0], 13)
        assert_equal(mod.exog.shape[0], 13)
        assert_equal(mod.sigma.shape, (13, 13))