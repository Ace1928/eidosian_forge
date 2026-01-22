import warnings
from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ridge import (
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from sklearn.model_selection import (
from sklearn.preprocessing import minmax_scale
from sklearn.utils import _IS_32BIT, check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def _make_sparse_offset_regression(n_samples=100, n_features=100, proportion_nonzero=0.5, n_informative=10, n_targets=1, bias=13.0, X_offset=30.0, noise=30.0, shuffle=True, coef=False, positive=False, random_state=None):
    X, y, c = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_targets=n_targets, bias=bias, noise=noise, shuffle=shuffle, coef=True, random_state=random_state)
    if n_features == 1:
        c = np.asarray([c])
    X += X_offset
    mask = np.random.RandomState(random_state).binomial(1, proportion_nonzero, X.shape) > 0
    removed_X = X.copy()
    X[~mask] = 0.0
    removed_X[mask] = 0.0
    y -= removed_X.dot(c)
    if positive:
        y += X.dot(np.abs(c) + 1 - c)
        c = np.abs(c) + 1
    if n_features == 1:
        c = c[0]
    if coef:
        return (X, y, c)
    return (X, y)