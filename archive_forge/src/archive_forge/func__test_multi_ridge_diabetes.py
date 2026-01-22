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
def _test_multi_ridge_diabetes(sparse_container):
    X = X_diabetes if sparse_container is None else sparse_container(X_diabetes)
    Y = np.vstack((y_diabetes, y_diabetes)).T
    n_features = X_diabetes.shape[1]
    ridge = Ridge(fit_intercept=False)
    ridge.fit(X, Y)
    assert ridge.coef_.shape == (2, n_features)
    Y_pred = ridge.predict(X)
    ridge.fit(X, y_diabetes)
    y_pred = ridge.predict(X)
    assert_array_almost_equal(np.vstack((y_pred, y_pred)).T, Y_pred, decimal=3)