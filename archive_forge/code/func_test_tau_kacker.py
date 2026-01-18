import io
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose
from statsmodels.regression.linear_model import WLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.stats.meta_analysis import (
from .results import results_meta
def test_tau_kacker(self):
    eff, var_eff = (self.eff, self.var_eff)
    t_PM, t_CA, t_DL, t_C2 = [0.8399, 1.1837, 0.5359, 0.9352]
    tau2, converged = _fit_tau_iterative(eff, var_eff, tau2_start=0.1, atol=1e-08)
    assert_equal(converged, True)
    assert_allclose(np.sqrt(tau2), t_PM, atol=6e-05)
    k = len(eff)
    tau2_ca = _fit_tau_mm(eff, var_eff, np.ones(k) / k)
    assert_allclose(np.sqrt(tau2_ca), t_CA, atol=6e-05)
    tau2_dl = _fit_tau_mm(eff, var_eff, 1 / var_eff)
    assert_allclose(np.sqrt(tau2_dl), t_DL, atol=0.001)
    tau2_dl_, converged = _fit_tau_iter_mm(eff, var_eff, tau2_start=0, maxiter=1)
    assert_equal(converged, False)
    assert_allclose(tau2_dl_, tau2_dl, atol=1e-10)
    tau2_c2, converged = _fit_tau_iter_mm(eff, var_eff, tau2_start=tau2_ca, maxiter=1)
    assert_equal(converged, False)
    assert_allclose(np.sqrt(tau2_c2), t_C2, atol=6e-05)