import warnings
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.sandbox.regression.penalized import TheilGLS
from statsmodels.base._penalized import PenalizedMixin
import statsmodels.base._penalties as smpen
class TestPenalizedGLMBinomCountNoPenal(CheckPenalizedBinomCount):

    @classmethod
    def _initialize(cls):
        y, x = (cls.y, cls.x)
        x = x[:, :4]
        offset = -0.25 * np.ones(len(y))
        modp = GLM(y, x, family=family.Binomial(), offset=offset)
        cls.res2 = modp.fit(method='bfgs', max_start_irls=100)
        mod = GLMPenalized(y, x, family=family.Binomial(), offset=offset, penal=cls.penalty)
        mod.pen_weight = 0
        cls.res1 = mod.fit(method='bfgs', max_start_irls=3, maxiter=100, disp=0, start_params=cls.res2.params * 0.9)
        cls.atol = 1e-10
        cls.k_params = 4

    def test_deriv(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.model.score(res2.params * 0.98), res2.model.score(res2.params * 0.98), rtol=1e-10)
        assert_allclose(res1.model.score_obs(res2.params * 0.98), res2.model.score_obs(res2.params * 0.98), rtol=1e-10)