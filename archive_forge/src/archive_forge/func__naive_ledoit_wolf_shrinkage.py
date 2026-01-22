import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import (
from sklearn.covariance._shrunk_covariance import _ledoit_wolf
from sklearn.utils._testing import (
from .._shrunk_covariance import _oas
def _naive_ledoit_wolf_shrinkage(X):
    n_samples, n_features = X.shape
    emp_cov = empirical_covariance(X, assume_centered=False)
    mu = np.trace(emp_cov) / n_features
    delta_ = emp_cov.copy()
    delta_.flat[::n_features + 1] -= mu
    delta = (delta_ ** 2).sum() / n_features
    X2 = X ** 2
    beta_ = 1.0 / (n_features * n_samples) * np.sum(np.dot(X2.T, X2) / n_samples - emp_cov ** 2)
    beta = min(beta_, delta)
    shrinkage = beta / delta
    return shrinkage