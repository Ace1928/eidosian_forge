from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
class TestGeneralizedPoisson_p1:

    @classmethod
    def setup_class(cls):
        cls.data = load_randhie()
        cls.data.exog = sm.add_constant(cls.data.exog, prepend=False)
        cls.res1 = GeneralizedPoisson(cls.data.endog, cls.data.exog, p=1).fit(method='newton', disp=0)

    def test_llf(self):
        poisson_llf = sm.Poisson(self.data.endog, self.data.exog).loglike(self.res1.params[:-1])
        genpoisson_llf = sm.GeneralizedPoisson(self.data.endog, self.data.exog, p=1).loglike(list(self.res1.params[:-1]) + [0])
        assert_allclose(genpoisson_llf, poisson_llf)

    def test_score(self):
        poisson_score = sm.Poisson(self.data.endog, self.data.exog).score(self.res1.params[:-1])
        genpoisson_score = sm.GeneralizedPoisson(self.data.endog, self.data.exog, p=1).score(list(self.res1.params[:-1]) + [0])
        assert_allclose(genpoisson_score[:-1], poisson_score, atol=1e-09)

    def test_hessian(self):
        poisson_score = sm.Poisson(self.data.endog, self.data.exog).hessian(self.res1.params[:-1])
        genpoisson_score = sm.GeneralizedPoisson(self.data.endog, self.data.exog, p=1).hessian(list(self.res1.params[:-1]) + [0])
        assert_allclose(genpoisson_score[:-1, :-1], poisson_score, atol=1e-10)

    def test_t(self):
        unit_matrix = np.identity(self.res1.params.size)
        t_test = self.res1.t_test(unit_matrix)
        assert_allclose(self.res1.tvalues, t_test.tvalue)

    def test_fit_regularized(self):
        model = self.res1.model
        alpha = np.ones(len(self.res1.params))
        alpha[-2:] = 0
        res_reg1 = model.fit_regularized(alpha=alpha * 0.01, disp=0)
        res_reg2 = model.fit_regularized(alpha=alpha * 100, disp=0)
        res_reg3 = model.fit_regularized(alpha=alpha * 1000, disp=0)
        assert_allclose(res_reg1.params, self.res1.params, atol=5e-05)
        assert_allclose(res_reg1.bse, self.res1.bse, atol=1e-05)
        assert_allclose((self.res1.params[:-2] ** 2).mean(), 0.01658095554332078, rtol=1e-05)
        assert_allclose((res_reg1.params[:-2] ** 2).mean(), 0.016580734975068664, rtol=1e-05)
        assert_allclose((res_reg2.params[:-2] ** 2).mean(), 0.010672558641545994, rtol=1e-05)
        assert_allclose((res_reg3.params[:-2] ** 2).mean(), 0.00035544919793048415, rtol=1e-05)

    def test_init_kwds(self):
        kwds = self.res1.model._get_init_kwds()
        assert_('p' in kwds)
        assert_equal(kwds['p'], 1)

    def test_distr(self):
        check_distr(self.res1)