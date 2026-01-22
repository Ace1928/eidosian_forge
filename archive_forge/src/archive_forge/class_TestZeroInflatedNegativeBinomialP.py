from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
class TestZeroInflatedNegativeBinomialP(CheckGeneric):

    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        cls.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        exog = sm.add_constant(data.exog[:, 1], prepend=False)
        exog_infl = sm.add_constant(data.exog[:, 0], prepend=False)
        sp = np.array([1.88, -10.28, -0.2, 1.14, 1.34])
        cls.res1 = sm.ZeroInflatedNegativeBinomialP(data.endog, exog, exog_infl=exog_infl, p=2).fit(start_params=sp, method='nm', xtol=1e-06, maxiter=5000, disp=False)
        cls.res1._results._attach_nullmodel = True
        cls.init_keys = ['exog_infl', 'exposure', 'inflation', 'offset', 'p']
        cls.init_kwds = {'inflation': 'logit', 'p': 2}
        res2 = RandHIE.zero_inflated_negative_binomial
        cls.res2 = res2

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, atol=0.001, rtol=0.001)

    def test_conf_int(self):
        pass

    def test_bic(self):
        pass

    def test_fit_regularized(self):
        model = self.res1.model
        alpha = np.ones(len(self.res1.params))
        alpha[-2:] = 0
        res_reg = model.fit_regularized(alpha=alpha * 0.01, disp=False, maxiter=500)
        assert_allclose(res_reg.params[2:], self.res1.params[2:], atol=0.1, rtol=0.1)

    def test_minimize(self, reset_randomstate):
        model = self.res1.model
        start_params = self.res1.mle_settings['start_params']
        res_ncg = model.fit(start_params=start_params, method='minimize', min_method='trust-ncg', maxiter=500, disp=False)
        assert_allclose(res_ncg.params, self.res2.params, atol=0.001, rtol=0.03)
        assert_allclose(res_ncg.bse, self.res2.bse, atol=0.001, rtol=0.06)
        assert_(res_ncg.mle_retvals['converged'] is True)
        res_dog = model.fit(start_params=start_params, method='minimize', min_method='dogleg', maxiter=500, disp=False)
        assert_allclose(res_dog.params, self.res2.params, atol=0.001, rtol=0.003)
        assert_allclose(res_dog.bse, self.res2.bse, atol=0.001, rtol=0.007)
        assert_(res_dog.mle_retvals['converged'] is True)
        res_bh = model.fit(start_params=start_params, method='basinhopping', maxiter=500, niter_success=3, disp=False)
        assert_allclose(res_bh.params, self.res2.params, atol=0.0001, rtol=0.0003)
        assert_allclose(res_bh.bse, self.res2.bse, atol=0.001, rtol=0.001)