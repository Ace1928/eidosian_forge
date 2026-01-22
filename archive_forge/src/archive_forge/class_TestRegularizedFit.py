from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
class TestRegularizedFit:

    def test_empty_model(self):
        np.random.seed(742)
        n = 100
        endog = np.random.normal(size=n)
        exog = np.random.normal(size=(n, 3))
        for cls in (OLS, WLS, GLS):
            model = cls(endog, exog)
            result = model.fit_regularized(alpha=1000)
            assert_equal(result.params, 0.0)

    def test_regularized(self):
        import os
        from .results import glmnet_r_results
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        data = np.loadtxt(os.path.join(cur_dir, 'results', 'lasso_data.csv'), delimiter=',')
        tests = [x for x in dir(glmnet_r_results) if x.startswith('rslt_')]
        for test in tests:
            vec = getattr(glmnet_r_results, test)
            n = vec[0]
            p = vec[1]
            L1_wt = float(vec[2])
            lam = float(vec[3])
            params = vec[4:].astype(np.float64)
            endog = data[0:int(n), 0]
            exog = data[0:int(n), 1:int(p) + 1]
            endog = endog - endog.mean()
            endog /= endog.std(ddof=1)
            exog = exog - exog.mean(0)
            exog /= exog.std(0, ddof=1)
            for cls in (OLS, WLS, GLS):
                mod = cls(endog, exog)
                rslt = mod.fit_regularized(L1_wt=L1_wt, alpha=lam)
                assert_almost_equal(rslt.params, params, decimal=3)
                mod.fit_regularized(L1_wt=L1_wt, alpha=lam, profile_scale=True)

    def test_regularized_weights(self):
        np.random.seed(1432)
        exog1 = np.random.normal(size=(100, 3))
        endog1 = exog1[:, 0] + exog1[:, 1] + np.random.normal(size=100)
        exog2 = np.random.normal(size=(100, 3))
        endog2 = exog2[:, 0] + exog2[:, 1] + np.random.normal(size=100)
        exog_a = np.vstack((exog1, exog1, exog2))
        endog_a = np.concatenate((endog1, endog1, endog2))
        exog_b = np.vstack((exog1, exog2))
        endog_b = np.concatenate((endog1, endog2))
        wgts = np.ones(200)
        wgts[0:100] = 2
        sigma = np.diag(1 / wgts)
        for L1_wt in (0, 0.5, 1):
            for alpha in (0, 1):
                mod1 = OLS(endog_a, exog_a)
                rslt1 = mod1.fit_regularized(L1_wt=L1_wt, alpha=alpha)
                mod2 = WLS(endog_b, exog_b, weights=wgts)
                rslt2 = mod2.fit_regularized(L1_wt=L1_wt, alpha=alpha)
                mod3 = GLS(endog_b, exog_b, sigma=sigma)
                rslt3 = mod3.fit_regularized(L1_wt=L1_wt, alpha=alpha)
                assert_almost_equal(rslt1.params, rslt2.params, decimal=3)
                assert_almost_equal(rslt1.params, rslt3.params, decimal=3)

    def test_regularized_weights_list(self):
        np.random.seed(132)
        exog1 = np.random.normal(size=(100, 3))
        endog1 = exog1[:, 0] + exog1[:, 1] + np.random.normal(size=100)
        exog2 = np.random.normal(size=(100, 3))
        endog2 = exog2[:, 0] + exog2[:, 1] + np.random.normal(size=100)
        exog_a = np.vstack((exog1, exog1, exog2))
        endog_a = np.concatenate((endog1, endog1, endog2))
        exog_b = np.vstack((exog1, exog2))
        endog_b = np.concatenate((endog1, endog2))
        wgts = np.ones(200)
        wgts[0:100] = 2
        sigma = np.diag(1 / wgts)
        for L1_wt in (0, 0.5, 1):
            for alpha_element in (0, 1):
                alpha = [alpha_element] * 3
                mod1 = OLS(endog_a, exog_a)
                rslt1 = mod1.fit_regularized(L1_wt=L1_wt, alpha=alpha)
                mod2 = WLS(endog_b, exog_b, weights=wgts)
                rslt2 = mod2.fit_regularized(L1_wt=L1_wt, alpha=alpha)
                mod3 = GLS(endog_b, exog_b, sigma=sigma)
                rslt3 = mod3.fit_regularized(L1_wt=L1_wt, alpha=alpha)
                assert_almost_equal(rslt1.params, rslt2.params, decimal=3)
                assert_almost_equal(rslt1.params, rslt3.params, decimal=3)