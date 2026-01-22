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
class CheckPenalizedPoisson:

    @classmethod
    def setup_class(cls):
        np.random.seed(987865)
        nobs, k_vars = (500, 10)
        k_nonzero = 4
        x = (np.random.rand(nobs, k_vars) + 0.5 * (np.random.rand(nobs, 1) - 0.5)) * 2 - 1
        x *= 1.2
        x[:, 0] = 1
        beta = np.zeros(k_vars)
        beta[:k_nonzero] = 1.0 / np.arange(1, k_nonzero + 1)
        linpred = x.dot(beta)
        y = cls._generate_endog(linpred)
        cls.k_nonzero = k_nonzero
        cls.x = x
        cls.y = y
        cls.rtol = 0.0001
        cls.atol = 1e-06
        cls.exog_index = slice(None, None, None)
        cls.k_params = k_vars
        cls.skip_hessian = False
        cls.penalty = smpen.SCADSmoothed(0.1, c0=0.0001)
        cls._initialize()

    @classmethod
    def _generate_endog(cls, linpred):
        mu = np.exp(linpred)
        np.random.seed(999)
        y = np.random.poisson(mu)
        return y

    def test_params_table(self):
        res1 = self.res1
        res2 = self.res2
        assert_equal((res1.params != 0).sum(), self.k_params)
        assert_allclose(res1.params[self.exog_index], res2.params, rtol=self.rtol, atol=self.atol)
        assert_allclose(res1.bse[self.exog_index], res2.bse, rtol=self.rtol, atol=self.atol)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            assert_allclose(res1.pvalues[self.exog_index], res2.pvalues, rtol=self.rtol, atol=self.atol)
        assert_allclose(res1.predict(), res2.predict(), rtol=0.05)

    @pytest.mark.smoke
    def test_summary(self):
        self.res1.summary()

    @pytest.mark.smoke
    def test_summary2(self):
        summ = self.res1.summary2()
        assert isinstance(summ.as_latex(), str)
        assert isinstance(summ.as_html(), str)
        assert isinstance(summ.as_text(), str)

    def test_numdiff(self):
        res1 = self.res1
        p = res1.params * 0.98
        kwds = {'scale': 1} if isinstance(res1.model, GLM) else {}
        assert_allclose(res1.model.score(p, **kwds)[self.exog_index], res1.model.score_numdiff(p, **kwds)[self.exog_index], rtol=0.025)
        if not self.skip_hessian:
            if isinstance(self.exog_index, slice):
                idx1 = idx2 = self.exog_index
            else:
                idx1 = self.exog_index[:, None]
                idx2 = self.exog_index
            h1 = res1.model.hessian(res1.params, **kwds)[idx1, idx2]
            h2 = res1.model.hessian_numdiff(res1.params, **kwds)[idx1, idx2]
            assert_allclose(h1, h2, rtol=0.02)