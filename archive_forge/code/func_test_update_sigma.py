from math import log
import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_update_sigma(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    n_samples = n_features = 10
    X = rng.randn(n_samples, n_features)
    alpha = 1
    lmbda = np.arange(1, n_features + 1)
    keep_lambda = np.array([True] * n_features)
    reg = ARDRegression()
    sigma = reg._update_sigma(X, alpha, lmbda, keep_lambda)
    sigma_woodbury = reg._update_sigma_woodbury(X, alpha, lmbda, keep_lambda)
    np.testing.assert_allclose(sigma, sigma_woodbury)