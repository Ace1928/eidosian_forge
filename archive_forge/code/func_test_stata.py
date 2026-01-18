import os
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises
import pandas as pd
import pytest
from scipy.stats import norm
from statsmodels.datasets import macrodata
from statsmodels.genmod.api import GLM
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.stats.diagnostic import recursive_olsresiduals
from statsmodels.tools import add_constant
from statsmodels.tools.eval_measures import aic, bic
from statsmodels.tools.sm_exceptions import ValueWarning
def test_stata():
    mod = RecursiveLS(endog, exog, loglikelihood_burn=3)
    with pytest.warns(UserWarning):
        res = mod.fit()
    d = max(res.nobs_diffuse, res.loglikelihood_burn)
    assert_allclose(res.resid_recursive[3:], results_stata.iloc[3:]['rr'], atol=1e-05, rtol=1e-05)
    assert_allclose(res.cusum, results_stata.iloc[3:]['cusum'], atol=1e-05)
    assert_allclose(res.cusum_squares, results_stata.iloc[3:]['cusum2'], atol=1e-05)
    actual_bounds = res._cusum_significance_bounds(alpha=0.05, ddof=0, points=np.arange(d + 1, res.nobs + 1))
    desired_bounds = results_stata.iloc[3:][['lw', 'uw']].T
    assert_allclose(actual_bounds, desired_bounds, atol=1e-05)
    actual_bounds = res._cusum_squares_significance_bounds(alpha=0.05, points=np.arange(d + 1, res.nobs + 1))
    desired_bounds = results_stata.iloc[3:][['lww', 'uww']].T
    assert_allclose(actual_bounds, desired_bounds, atol=0.01)