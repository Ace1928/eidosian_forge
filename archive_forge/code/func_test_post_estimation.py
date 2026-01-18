import itertools
import os
import numpy as np
from statsmodels.duration.hazard_regression import PHReg
from numpy.testing import (assert_allclose,
import pandas as pd
import pytest
from .results import survival_r_results
from .results import survival_enet_r_results
def test_post_estimation(self):
    np.random.seed(34234)
    time = 50 * np.random.uniform(size=200)
    status = np.random.randint(0, 2, 200).astype(np.float64)
    exog = np.random.normal(size=(200, 4))
    mod = PHReg(time, exog, status)
    rslt = mod.fit()
    mart_resid = rslt.martingale_residuals
    assert_allclose(np.abs(mart_resid).sum(), 120.72475743348433)
    w_avg = rslt.weighted_covariate_averages
    assert_allclose(np.abs(w_avg[0]).sum(0), np.r_[7.31008415, 9.77608674, 10.89515885, 13.1106801])
    bc_haz = rslt.baseline_cumulative_hazard
    v = [np.mean(np.abs(x)) for x in bc_haz[0]]
    w = np.r_[23.482841556421608, 0.44149255358417017, 0.6866011408127528]
    assert_allclose(v, w)
    score_resid = rslt.score_residuals
    v = np.r_[0.50924792, 0.4533952, 0.4876718, 0.5441128]
    w = np.abs(score_resid).mean(0)
    assert_allclose(v, w)
    groups = np.random.randint(0, 3, 200)
    mod = PHReg(time, exog, status)
    rslt = mod.fit(groups=groups)
    robust_cov = rslt.cov_params()
    v = [0.00513432, 0.01278423, 0.00810427, 0.00293147]
    w = np.abs(robust_cov).mean(0)
    assert_allclose(v, w, rtol=1e-06)
    s_resid = rslt.schoenfeld_residuals
    ii = np.flatnonzero(np.isfinite(s_resid).all(1))
    s_resid = s_resid[ii, :]
    v = np.r_[0.85154336, 0.72993748, 0.73758071, 0.78599333]
    assert_allclose(np.abs(s_resid).mean(0), v)