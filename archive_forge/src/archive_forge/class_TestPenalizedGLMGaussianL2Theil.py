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
class TestPenalizedGLMGaussianL2Theil(CheckPenalizedGaussian):

    @classmethod
    def _initialize(cls):
        y, x = (cls.y, cls.x)
        y = y + 10
        k = x.shape[1]
        cov_type = 'HC0'
        restriction = np.eye(k)[2:]
        modp = TheilGLS(y, x, r_matrix=restriction)
        cls.res2 = modp.fit(pen_weight=120.74564413221599 * 1000, use_t=False)
        pen = smpen.L2ConstraintsPenalty(restriction=restriction)
        mod = GLMPenalized(y, x, family=family.Gaussian(), penal=pen)
        mod.pen_weight *= 1
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0, trim=False)
        cls.k_nonzero = k
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.k_params = x.shape[1]
        cls.atol = 1e-05
        cls.rtol = 1e-05

    def test_params_table(self):
        res1 = self.res1
        res2 = self.res2
        assert_equal((res1.params != 0).sum(), self.k_params)
        assert_allclose(res1.params, res2.params, rtol=self.rtol, atol=self.atol)
        exog_index = slice(None, None, None)
        assert_allclose(res1.bse[exog_index], res2.bse[exog_index], rtol=0.1, atol=self.atol)
        assert_allclose(res1.tvalues[exog_index], res2.tvalues[exog_index], rtol=0.08, atol=0.005)
        assert_allclose(res1.pvalues[exog_index], res2.pvalues[exog_index], rtol=0.1, atol=0.005)
        assert_allclose(res1.predict(), res2.predict(), rtol=1e-05)