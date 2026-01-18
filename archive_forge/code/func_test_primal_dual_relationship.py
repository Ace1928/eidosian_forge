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
def test_primal_dual_relationship():
    y = y_diabetes.reshape(-1, 1)
    coef = _solve_cholesky(X_diabetes, y, alpha=[0.01])
    K = np.dot(X_diabetes, X_diabetes.T)
    dual_coef = _solve_cholesky_kernel(K, y, alpha=[0.01])
    coef2 = np.dot(X_diabetes.T, dual_coef).T
    assert_array_almost_equal(coef, coef2)