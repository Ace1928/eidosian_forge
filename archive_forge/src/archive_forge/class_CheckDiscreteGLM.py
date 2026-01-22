import os
import numpy as np
import pandas as pd
import pytest
import statsmodels.discrete.discrete_model as smd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.regression.linear_model import OLS
from statsmodels.base.covtype import get_robustcov_results
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import add_constant
from numpy.testing import assert_allclose, assert_equal, assert_
import statsmodels.tools._testing as smt
from .results import results_count_robust_cluster as results_st
class CheckDiscreteGLM:

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert_equal(res1.cov_type, self.cov_type)
        assert_equal(res2.cov_type, self.cov_type)
        rtol = getattr(res1, 'rtol', 1e-13)
        assert_allclose(res1.params, res2.params, rtol=rtol)
        assert_allclose(res1.bse, res2.bse, rtol=1e-10)

    def test_score_hessian(self):
        res1 = self.res1
        res2 = self.res2
        if isinstance(res2.model, OLS):
            kwds = {'scale': res2.scale}
        else:
            kwds = {}
        if isinstance(res2.model, OLS):
            sgn = +1
        else:
            sgn = -1
        score1 = res1.model.score(res1.params * 0.98, scale=res1.scale)
        score2 = res2.model.score(res1.params * 0.98, **kwds)
        assert_allclose(score1, score2, rtol=1e-13)
        hess1 = res1.model.hessian(res1.params, scale=res1.scale)
        hess2 = res2.model.hessian(res1.params, **kwds)
        assert_allclose(hess1, hess2, rtol=1e-10)
        if isinstance(res2.model, OLS):
            return
        scoref1 = res1.model.score_factor(res1.params, scale=res1.scale)
        scoref2 = res2.model.score_factor(res1.params, **kwds)
        assert_allclose(scoref1, scoref2, rtol=1e-10)
        hessf1 = res1.model.hessian_factor(res1.params, scale=res1.scale)
        hessf2 = res2.model.hessian_factor(res1.params, **kwds)
        assert_allclose(sgn * hessf1, hessf2, rtol=1e-10)

    def test_score_test(self):
        res1 = self.res1
        res2 = self.res2
        if isinstance(res2.model, OLS):
            return
        fitted = self.res1.fittedvalues
        exog_extra = np.column_stack((fitted ** 2, fitted ** 3))
        res_lm1 = res1.score_test(exog_extra, cov_type='nonrobust')
        res_lm2 = res2.score_test(exog_extra, cov_type='nonrobust')
        assert_allclose(np.hstack(res_lm1), np.hstack(res_lm2), rtol=5e-07)

    def test_margeff(self):
        if isinstance(self.res2.model, OLS) or hasattr(self.res1.model, 'offset'):
            pytest.skip('not available yet')
        marg1 = self.res1.get_margeff()
        marg2 = self.res2.get_margeff()
        assert_allclose(marg1.margeff, marg2.margeff, rtol=1e-10)
        assert_allclose(marg1.margeff_se, marg2.margeff_se, rtol=1e-10)
        marg1 = self.res1.get_margeff(count=True, dummy=True)
        marg2 = self.res2.get_margeff(count=True, dummy=True)
        assert_allclose(marg1.margeff, marg2.margeff, rtol=1e-10)
        assert_allclose(marg1.margeff_se, marg2.margeff_se, rtol=1e-10)