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
class TestGlmBinomial(CheckModelResultsMixin):

    @classmethod
    def setup_class(cls):
        """
        Test Binomial family with canonical logit link using star98 dataset.
        """
        cls.decimal_resids = DECIMAL_1
        cls.decimal_bic = DECIMAL_2
        from statsmodels.datasets.star98 import load
        from .results.results_glm import Star98
        data = load()
        data.endog = np.require(data.endog, requirements='W')
        data.exog = np.require(data.exog, requirements='W')
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = GLM(data.endog, data.exog, family=sm.families.Binomial()).fit()
        cls.res2 = Star98()

    def test_endog_dtype(self):
        from statsmodels.datasets.star98 import load
        data = load()
        data.exog = add_constant(data.exog, prepend=False)
        endog = data.endog.astype(int)
        res2 = GLM(endog, data.exog, family=sm.families.Binomial()).fit()
        assert_allclose(res2.params, self.res1.params)
        endog = data.endog.astype(np.double)
        res3 = GLM(endog, data.exog, family=sm.families.Binomial()).fit()
        assert_allclose(res3.params, self.res1.params)

    def test_invalid_endog(self, reset_randomstate):
        endog = np.random.randint(0, 100, size=(1000, 3))
        exog = np.random.standard_normal((1000, 2))
        with pytest.raises(ValueError, match='endog has more than 2 columns'):
            GLM(endog, exog, family=sm.families.Binomial())

    def test_invalid_endog_formula(self, reset_randomstate):
        n = 200
        exog = np.random.normal(size=(n, 2))
        endog = np.random.randint(0, 3, size=n).astype(str)
        data = pd.DataFrame({'y': endog, 'x1': exog[:, 0], 'x2': exog[:, 1]})
        with pytest.raises(ValueError, match='array with multiple columns'):
            sm.GLM.from_formula('y ~ x1 + x2', data, family=sm.families.Binomial())

    def test_get_distribution_binom_count(self):
        res1 = self.res1
        res_scale = 1
        mu_prob = res1.fittedvalues
        n = res1.model.n_trials
        distr = res1.model.family.get_distribution(mu_prob, res_scale, n_trials=n)
        var_endog = res1.model.family.variance(mu_prob) * res_scale
        m, v = distr.stats()
        assert_allclose(mu_prob * n, m, rtol=1e-13)
        assert_allclose(var_endog * n, v, rtol=1e-13)
        distr2 = res1.model.get_distribution(res1.params, res_scale, n_trials=n)
        for k in distr2.kwds:
            assert_allclose(distr.kwds[k], distr2.kwds[k], rtol=1e-13)