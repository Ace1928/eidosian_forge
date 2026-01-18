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
def test_regularized_gaussian():
    np.random.seed(8735)
    ng, gs, p = (200, 4, 200)
    groups = np.kron(np.arange(ng), np.ones(gs))
    x = np.zeros((ng * gs, p))
    x[:, 0] = 1 * (np.random.uniform(size=ng * gs) < 0.5)
    x[:, 1] = np.random.normal(size=ng * gs)
    r = 0.5
    for j in range(2, p):
        eps = np.random.normal(size=ng * gs)
        x[:, j] = r * x[:, j - 1] + np.sqrt(1 - r ** 2) * eps
    lpr = np.dot(x[:, 0:4], np.r_[2, 3, 1.5, 2])
    s = 0.4
    e = np.sqrt(s) * np.kron(np.random.normal(size=ng), np.ones(gs))
    e += np.sqrt(1 - s) * np.random.normal(size=ng * gs)
    y = lpr + e
    model = gee.GEE(y, x, cov_struct=cov_struct.Exchangeable(), groups=groups)
    result = model.fit_regularized(0.01, maxiter=100)
    ex = np.zeros(200)
    ex[0:4] = np.r_[2, 3, 1.5, 2]
    assert_allclose(result.params, ex, rtol=0.01, atol=0.2)
    assert_allclose(model.cov_struct.dep_params, np.r_[s], rtol=0.01, atol=0.05)