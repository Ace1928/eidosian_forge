from statsmodels.compat.python import lrange, lmap
import os
import copy
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
import statsmodels.sandbox.regression.gmm as gmm
class TestGMMSt2:

    @classmethod
    def setup_class(cls):
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=2, inv_weights=w0inv, wargs={'ddof': 0, 'centered': False}, optim_method='bfgs', optim_args={'gtol': 1e-06, 'disp': 0})
        cls.res1 = res
        from .results_ivreg2_griliches import results_gmm2s_robust as results
        cls.res2 = results
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=1, inv_weights=w0inv, wargs={'ddof': 0, 'centered': False}, optim_method='bfgs', optim_args={'gtol': 1e-06, 'disp': 0})
        cls.res3 = res

    def test_basic(self):
        res1, res2 = (self.res1, self.res2)
        assert_allclose(res1.params, res2.params, rtol=5e-05, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=5e-06)
        n = res1.model.exog.shape[0]
        dffact = np.sqrt(745.0 / 758)
        assert_allclose(res1.bse * dffact, res2.bse, rtol=0.005, atol=0)
        assert_allclose(res1.bse * dffact, res2.bse, rtol=0, atol=0.005)
        bse = np.sqrt(np.diag(res1._cov_params(has_optimal_weights=True, weights=res1.weights)))
        assert_allclose(res1.bse, res2.bse, rtol=0.5, atol=0)
        bse = np.sqrt(np.diag(res1._cov_params(has_optimal_weights=True, weights=res1.weights, use_weights=True)))
        assert_allclose(res1.bse, res2.bse, rtol=0.05, atol=0)
        assert_allclose(self.res3.bse, res2.bse, rtol=5e-05, atol=0)
        assert_allclose(self.res3.bse, res2.bse, rtol=0, atol=5e-06)