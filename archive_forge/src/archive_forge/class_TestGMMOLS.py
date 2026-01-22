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
class TestGMMOLS:

    @classmethod
    def setup_class(cls):
        exog = exog_st
        res_ols = OLS(endog, exog).fit()
        nobs, k_instr = exog.shape
        w0inv = np.dot(exog.T, exog) / nobs
        mod = gmm.IVGMM(endog, exog, exog)
        res = mod.fit(np.ones(exog.shape[1], float), maxiter=0, inv_weights=w0inv, optim_method='bfgs', optim_args={'gtol': 1e-06, 'disp': 0})
        cls.res1 = res
        cls.res2 = res_ols

    def test_basic(self):
        res1, res2 = (self.res1, self.res2)
        assert_allclose(res1.params, res2.params, rtol=0.0005, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=1e-05)
        n = res1.model.exog.shape[0]
        dffac = 1
        assert_allclose(res1.bse * dffac, res2.HC0_se, rtol=5e-06, atol=0)
        assert_allclose(res1.bse * dffac, res2.HC0_se, rtol=0, atol=1e-07)

    @pytest.mark.xfail(reason='Not asserting anything meaningful', raises=NotImplementedError, strict=True)
    def test_other(self):
        res1, res2 = (self.res1, self.res2)
        raise NotImplementedError