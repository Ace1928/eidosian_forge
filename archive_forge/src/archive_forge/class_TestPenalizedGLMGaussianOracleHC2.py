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
class TestPenalizedGLMGaussianOracleHC2(CheckPenalizedGaussian):

    @classmethod
    def _initialize(cls):
        y, x = (cls.y, cls.x)
        y = y + 10
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Gaussian())
        cls.res2 = modp.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0)
        mod = GLMPenalized(y, x, family=family.Gaussian(), penal=cls.penalty)
        mod.pen_weight *= 1.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0, trim=True)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.k_params = cls.k_nonzero
        cls.atol = 1e-05
        cls.rtol = 1e-05