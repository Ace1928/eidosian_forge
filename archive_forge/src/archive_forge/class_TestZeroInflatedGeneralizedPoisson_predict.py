from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
class TestZeroInflatedGeneralizedPoisson_predict:

    @classmethod
    def setup_class(cls):
        expected_params = [1, 0.5, 0.5]
        np.random.seed(999)
        nobs = 2000
        exog = np.ones((nobs, 2))
        exog[:nobs // 2, 1] = 2
        mu_true = exog.dot(expected_params[:-1])
        cls.endog = sm.distributions.zigenpoisson.rvs(mu_true, expected_params[-1], 2, 0.5, size=mu_true.shape)
        model = sm.ZeroInflatedGeneralizedPoisson(cls.endog, exog, p=2)
        cls.res = model.fit(method='bfgs', maxiter=5000, disp=False)
        cls.params_true = [mu_true, expected_params[-1], 2, 0.5, nobs]

    def test_mean(self):

        def compute_conf_interval_95(mu, alpha, p, prob_infl, nobs):
            p = p - 1
            dispersion_factor = (1 + alpha * mu ** p) ** 2 + prob_infl * mu
            var = (dispersion_factor * (1 - prob_infl) * mu).mean()
            var += (((1 - prob_infl) * mu) ** 2).mean()
            var -= ((1 - prob_infl) * mu).mean() ** 2
            std = np.sqrt(var)
            conf_intv_95 = 2 * std / np.sqrt(nobs)
            return conf_intv_95
        conf_interval_95 = compute_conf_interval_95(*self.params_true)
        assert_allclose(self.res.predict().mean(), self.endog.mean(), atol=conf_interval_95, rtol=0)

    def test_var(self):

        def compute_mixture_var(dispersion_factor, prob_main, mu):
            prob_infl = 1 - prob_main
            var = (dispersion_factor * (1 - prob_infl) * mu).mean()
            var += (((1 - prob_infl) * mu) ** 2).mean()
            var -= ((1 - prob_infl) * mu).mean() ** 2
            return var
        res = self.res
        var_fitted = compute_mixture_var(res._dispersion_factor, res.predict(which='prob-main'), res.predict(which='mean-main'))
        assert_allclose(var_fitted.mean(), self.endog.var(), atol=0.05, rtol=0.1)

    def test_predict_prob(self):
        res = self.res
        pr = res.predict(which='prob')
        pr2 = sm.distributions.zinegbin.pmf(np.arange(pr.shape[1])[:, None], res.predict(), 0.5, 2, 0.5).T
        assert_allclose(pr, pr2, rtol=0.08, atol=0.05)