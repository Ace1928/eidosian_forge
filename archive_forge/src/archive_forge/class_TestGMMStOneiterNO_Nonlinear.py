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
class TestGMMStOneiterNO_Nonlinear(CheckGMM):

    @classmethod
    def setup_class(cls):
        cls.params_tol = [5e-05, 5e-06]
        cls.bse_tol = [5e-06, 0.1]
        exog = exog_st
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs

        def func(params, exog):
            return np.dot(exog, params)
        mod = gmm.NonlinearIVGMM(endog, exog, instrument, func)
        res = mod.fit(start, maxiter=1, inv_weights=w0inv, optim_method='bfgs', optim_args={'gtol': 1e-08, 'disp': 0}, wargs={'centered': False}, has_optimal_weights=False)
        cls.res1 = res
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=1, inv_weights=w0inv, optim_method='bfgs', optim_args={'gtol': 1e-06, 'disp': 0}, wargs={'centered': False}, has_optimal_weights=False)
        cls.res3 = res
        from .results_gmm_griliches import results_onestep as results
        cls.res2 = results

    @pytest.mark.xfail(reason='q vs Q comparison fails', raises=AssertionError, strict=True)
    def test_other(self):
        super().test_other()

    def test_score(self):
        params = self.res1.params * 1.1
        weights = self.res1.weights
        sc1 = self.res1.model.score(params, weights)
        sc2 = super(self.res1.model.__class__, self.res1.model).score(params, weights)
        assert_allclose(sc1, sc2, rtol=1e-06, atol=0)
        assert_allclose(sc1, sc2, rtol=0, atol=1e-07)
        sc1 = self.res1.model.score(self.res1.params, weights)
        assert_allclose(sc1, np.zeros(len(params)), rtol=0, atol=1e-08)