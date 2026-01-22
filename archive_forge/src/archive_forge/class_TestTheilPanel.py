import os
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.sandbox.regression.penalized import TheilGLS
class TestTheilPanel:

    @classmethod
    def setup_class(cls):
        nobs = 300
        nobs_i = 5
        n_groups = nobs // nobs_i
        k_vars = 3
        from statsmodels.sandbox.panel.random_panel import PanelSample
        dgp = PanelSample(nobs, k_vars, n_groups, seed=303305)
        dgp.group_means = 2 + dgp.random_state.randn(n_groups)
        print('seed', dgp.seed)
        y = dgp.generate_panel()
        x = np.column_stack((dgp.exog[:, 1:], dgp.groups[:, None] == np.arange(n_groups)))
        cls.dgp = dgp
        cls.endog = y
        cls.exog = x
        cls.res_ols = OLS(y, x).fit()

    def test_regression(self):
        y = self.endog
        x = self.exog
        n_groups, k_vars = (self.dgp.n_groups, self.dgp.k_vars)
        Rg = np.eye(n_groups - 1) - 1.0 / n_groups * np.ones((n_groups - 1, n_groups - 1))
        R = np.c_[np.zeros((n_groups - 1, k_vars)), Rg]
        r = np.zeros(n_groups - 1)
        R[:, k_vars - 1] = -1
        lambd = 1
        mod = TheilGLS(y, x, r_matrix=R, q_matrix=r, sigma_prior=lambd)
        res = mod.fit()
        params1 = np.array([0.9751655, 1.05215277, 0.37135028, 2.0492626, 2.82062503, 2.82139775, 1.92940468, 2.96942081, 2.86349583, 3.20695368, 4.04516422, 3.04918839, 4.54748808, 3.49026961, 3.15529618, 4.25552932, 2.65471759, 3.62328747, 3.07283053, 3.49485898, 3.42301424, 2.94677593, 2.81549427, 2.24895113, 2.29222784, 2.89194946, 3.17052308, 2.37754241, 3.54358533, 3.79838425, 1.91189071, 1.15976407, 4.05629691, 1.58556827, 4.49941666, 4.08608599, 3.1889269, 2.86203652, 3.06785013, 1.9376162, 2.90657681, 3.71910592, 3.15607617, 3.58464547, 2.15466323, 4.87026717, 2.92909833, 2.64998337, 2.891171, 4.04422964, 3.54616122, 4.12135273, 3.70232028, 3.8314497, 2.2591451, 2.39321422, 3.13064532, 2.1569678, 2.04667506, 3.92064689, 3.66243644, 3.11742725])
        assert_allclose(res.params, params1)
        pen_weight_aicc = mod.select_pen_weight(method='aicc')
        pen_weight_gcv = mod.select_pen_weight(method='gcv')
        pen_weight_cv = mod.select_pen_weight(method='cv')
        pen_weight_bic = mod.select_pen_weight(method='bic')
        assert_allclose(pen_weight_gcv, pen_weight_aicc, rtol=0.1)
        assert_allclose(pen_weight_aicc, 4.77333984, rtol=0.0001)
        assert_allclose(pen_weight_gcv, 4.45546875, rtol=0.0001)
        assert_allclose(pen_weight_bic, 9.35957031, rtol=0.0001)
        assert_allclose(pen_weight_cv, 1.99277344, rtol=0.0001)

    def test_combine_subset_regression(self):
        endog = self.endog
        exog = self.exog
        nobs = len(endog)
        n05 = nobs // 2
        np.random.seed(987125)
        shuffle_idx = np.random.permutation(np.arange(nobs))
        ys = endog[shuffle_idx]
        xs = exog[shuffle_idx]
        k = 10
        res_ols0 = OLS(ys[:n05], xs[:n05, :k]).fit()
        res_ols1 = OLS(ys[n05:], xs[n05:, :k]).fit()
        w = res_ols1.scale / res_ols0.scale
        mod_1 = TheilGLS(ys[n05:], xs[n05:, :k], r_matrix=np.eye(k), q_matrix=res_ols0.params, sigma_prior=w * res_ols0.cov_params())
        res_1p = mod_1.fit(cov_type='data-prior')
        res_1s = mod_1.fit(cov_type='sandwich')
        res_olsf = OLS(ys, xs[:, :k]).fit()
        assert_allclose(res_1p.params, res_olsf.params, rtol=1e-09)
        corr_fact = np.sqrt(res_1p.scale / res_olsf.scale)
        assert_allclose(res_1p.bse, res_olsf.bse * corr_fact, rtol=0.001)
        bse1 = np.array([0.26589869, 0.15224812, 0.38407399, 0.75679949, 0.660842, 0.5417408, 0.53697607, 0.66006377, 0.38228551, 0.53920485])
        assert_allclose(res_1s.bse, bse1, rtol=1e-07)