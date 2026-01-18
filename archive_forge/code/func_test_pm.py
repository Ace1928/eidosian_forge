import io
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose
from statsmodels.regression.linear_model import WLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.stats.meta_analysis import (
from .results import results_meta
def test_pm(self):
    res = results_meta.exk1_metafor
    eff, var_eff = (self.eff, self.var_eff)
    tau2, converged = _fit_tau_iterative(eff, var_eff, tau2_start=0.1, atol=1e-08)
    assert_equal(converged, True)
    assert_allclose(tau2, res.tau2, atol=1e-10)
    mod_wls = WLS(eff, np.ones(len(eff)), weights=1 / (var_eff + tau2))
    res_wls = mod_wls.fit(cov_type='fixed_scale')
    assert_allclose(res_wls.params, res.b, atol=1e-13)
    assert_allclose(res_wls.bse, res.se, atol=1e-10)
    ci_low, ci_upp = res_wls.conf_int()[0]
    assert_allclose(ci_low, res.ci_lb, atol=1e-10)
    assert_allclose(ci_upp, res.ci_ub, atol=1e-10)
    res3 = combine_effects(eff, var_eff, method_re='pm', atol=1e-07)
    assert_allclose(res3.tau2, res.tau2, atol=1e-10)
    assert_allclose(res3.mean_effect_re, res.b, atol=1e-13)
    assert_allclose(res3.sd_eff_w_re, res.se, atol=1e-10)
    ci = res3.conf_int(use_t=False)[1]
    assert_allclose(ci[0], res.ci_lb, atol=1e-10)
    assert_allclose(ci[1], res.ci_ub, atol=1e-10)
    assert_allclose(res3.q, res.QE, atol=1e-10)
    th = res3.test_homogeneity()
    q, pv = th
    df = th.df
    assert_allclose(pv, res.QEp, atol=1e-10)
    assert_allclose(q, res.QE, atol=1e-10)
    assert_allclose(df, 9 - 1, atol=1e-10)