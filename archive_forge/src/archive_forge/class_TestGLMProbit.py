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
class TestGLMProbit(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        endog_bin = (endog > endog.mean()).astype(int)
        cls.cov_type = 'cluster'
        mod1 = GLM(endog_bin, exog, family=families.Binomial(link=links.Probit()))
        cls.res1 = mod1.fit(method='newton', cov_type='cluster', cov_kwds=dict(groups=group))
        mod1 = smd.Probit(endog_bin, exog)
        cls.res2 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))
        cls.rtol = 1e-06

    def test_score_hessian(self):
        res1 = self.res1
        res2 = self.res2
        score1 = res1.model.score(res1.params * 0.98)
        score2 = res2.model.score(res1.params * 0.98)
        assert_allclose(score1, score2, rtol=1e-13)
        hess1 = res1.model.hessian(res1.params)
        hess2 = res2.model.hessian(res1.params)
        assert_allclose(hess1, hess2, rtol=1e-13)