from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
def test_stationary_grid(self):
    endog = np.r_[4, 2, 3, 1, 4, 5, 6, 7, 8, 3, 2, 4.0]
    exog = np.r_[2, 3, 1, 4, 3, 2, 5, 4, 5, 6, 3, 2]
    group = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    exog = tools.add_constant(exog)
    cs = cov_struct.Stationary(max_lag=2, grid=True)
    model = gee.GEE(endog, exog, group, cov_struct=cs)
    result = model.fit()
    se = result.bse * np.sqrt(12 / 9.0)
    assert_allclose(cs.covariance_matrix(np.r_[1, 1, 1], 0)[0].sum(), 6.463353828514945)
    assert_allclose(result.params, np.r_[4.463968, -0.0386674], rtol=1e-05, atol=1e-05)
    assert_allclose(se, np.r_[0.5217202, 0.2800333], rtol=1e-05, atol=1e-05)