import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
class CheckWtdDuplicationMixin:
    decimal_params = DECIMAL_4

    @classmethod
    def setup_class(cls):
        cls.data = cpunish.load()
        cls.data.endog = np.asarray(cls.data.endog)
        cls.data.exog = np.asarray(cls.data.exog)
        cls.endog = cls.data.endog
        cls.exog = cls.data.exog
        np.random.seed(1234)
        cls.weight = np.random.randint(5, 100, len(cls.endog))
        cls.endog_big = np.repeat(cls.endog, cls.weight)
        cls.exog_big = np.repeat(cls.exog, cls.weight, axis=0)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, atol=1e-06, rtol=1e-06)
    decimal_bse = DECIMAL_4

    def test_standard_errors(self):
        assert_allclose(self.res1.bse, self.res2.bse, rtol=1e-05, atol=1e-06)
    decimal_resids = DECIMAL_4
    '\n    def test_residuals(self):\n        resids1 = np.column_stack((self.res1.resid_pearson,\n                                   self.res1.resid_deviance,\n                                   self.res1.resid_working,\n                                   self.res1.resid_anscombe,\n                                   self.res1.resid_response))\n        resids2 = np.column_stack((self.res1.resid_pearson,\n                                   self.res2.resid_deviance,\n                                   self.res2.resid_working,\n                                   self.res2.resid_anscombe,\n                                   self.res2.resid_response))\n        assert_allclose(resids1, resids2, self.decimal_resids)\n    '

    def test_aic(self):
        assert_allclose(self.res1.aic, self.res2.aic, atol=1e-06, rtol=1e-06)

    def test_deviance(self):
        assert_allclose(self.res1.deviance, self.res2.deviance, atol=1e-06, rtol=1e-06)

    def test_scale(self):
        assert_allclose(self.res1.scale, self.res2.scale, atol=1e-06, rtol=1e-06)

    def test_loglike(self):
        assert_allclose(self.res1.llf, self.res2.llf, 1e-06)
    decimal_null_deviance = DECIMAL_4

    def test_null_deviance(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DomainWarning)
            assert_allclose(self.res1.null_deviance, self.res2.null_deviance, atol=1e-06, rtol=1e-06)
    decimal_bic = DECIMAL_4

    def test_bic(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert_allclose(self.res1.bic, self.res2.bic, atol=1e-06, rtol=1e-06)
    decimal_fittedvalues = DECIMAL_4

    def test_fittedvalues(self):
        res2_fitted = self.res2.predict(self.res1.model.exog)
        assert_allclose(self.res1.fittedvalues, res2_fitted, atol=1e-05, rtol=1e-05)
    decimal_tpvalues = DECIMAL_4

    def test_tpvalues(self):
        assert_allclose(self.res1.tvalues, self.res2.tvalues, atol=1e-06, rtol=0.0002)
        assert_allclose(self.res1.pvalues, self.res2.pvalues, atol=1e-06, rtol=1e-06)
        assert_allclose(self.res1.conf_int(), self.res2.conf_int(), atol=1e-06, rtol=1e-06)