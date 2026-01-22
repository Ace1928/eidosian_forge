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
class TestPenalizedLogitOraclePenalized2(CheckPenalizedLogit):

    @classmethod
    def _initialize(cls):
        y, x = (cls.y, cls.x)
        modp = LogitPenalized(y, x[:, :cls.k_nonzero], penal=cls.penalty)
        modp.pen_weight *= 0.5
        modp.penal.tau = 0.05
        cls.res2 = modp.fit(method='bfgs', maxiter=100, disp=0)
        mod = LogitPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= 0.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, trim=True, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 1e-08
        cls.k_params = cls.k_nonzero

    def test_zeros(self):
        assert_equal(self.res1.params[self.k_nonzero:], 0)
        assert_equal(self.res1.bse[self.k_nonzero:], 0)