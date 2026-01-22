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
class TestPenalizedGLMBinomCountOracleHC(CheckPenalizedBinomCount):

    @classmethod
    def _initialize(cls):
        y, x = (cls.y, cls.x)
        offset = -0.25 * np.ones(len(y))
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Binomial(), offset=offset)
        cls.res2 = modp.fit(cov_type=cov_type, method='newton', maxiter=1000, disp=0)
        mod = GLMPenalized(y, x, family=family.Binomial(), offset=offset, penal=cls.penalty)
        mod.pen_weight *= 1
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', max_start_irls=0, maxiter=100, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 0.001