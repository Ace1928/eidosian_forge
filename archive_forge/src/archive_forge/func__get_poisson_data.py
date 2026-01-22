import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.base._screening import VariableScreening
def _get_poisson_data():
    np.random.seed(987865)
    nobs, k_vars = (100, 500)
    k_nonzero = 5
    x = (np.random.rand(nobs, k_vars) + 1.0 * (np.random.rand(nobs, 1) - 0.5)) * 2 - 1
    x *= 1.2
    x = (x - x.mean(0)) / x.std(0)
    x[:, 0] = 1
    beta = np.zeros(k_vars)
    idx_nonzero_true = [0, 100, 300, 400, 411]
    beta[idx_nonzero_true] = 1.0 / np.arange(1, k_nonzero + 1)
    beta = np.sqrt(beta)
    linpred = x.dot(beta)
    mu = np.exp(linpred)
    y = np.random.poisson(mu)
    return (y, x, idx_nonzero_true, beta)