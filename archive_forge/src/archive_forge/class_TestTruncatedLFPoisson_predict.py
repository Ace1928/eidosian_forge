import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels import datasets
from statsmodels.tools.tools import add_constant
from statsmodels.tools.testing import Holder
from statsmodels.tools.sm_exceptions import (
from statsmodels.distributions.discrete import (
from statsmodels.discrete.truncated_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results.results_discrete import RandHIE
from .results import results_truncated as results_t
from .results import results_truncated_st as results_ts
class TestTruncatedLFPoisson_predict:

    @classmethod
    def setup_class(cls):
        cls.expected_params = [1, 0.5]
        np.random.seed(123)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs // 2, 1] = 2
        mu_true = exog.dot(cls.expected_params)
        cls.endog = truncatedpoisson.rvs(mu_true, 0, size=mu_true.shape)
        model = TruncatedLFPoisson(cls.endog, exog, truncation=0)
        cls.res = model.fit(method='bfgs', maxiter=5000)

    def test_mean(self):
        assert_allclose(self.res.predict().mean(), self.endog.mean(), atol=0.2, rtol=0.2)

    def test_var(self):
        v = self.res.predict(which='var').mean()
        assert_allclose(v, self.endog.var(), atol=0.2, rtol=0.2)
        return
        assert_allclose(self.res.predict().mean() * self.res._dispersion_factor.mean(), self.endog.var(), atol=0.05, rtol=0.05)

    def test_predict_prob(self):
        res = self.res
        pr = res.predict(which='prob')
        pr2 = truncatedpoisson.pmf(np.arange(8), res.predict(which='mean-main')[:, None], 0)
        assert_allclose(pr, pr2, rtol=1e-10, atol=1e-10)