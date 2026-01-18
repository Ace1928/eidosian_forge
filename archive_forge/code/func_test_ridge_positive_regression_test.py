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
@pytest.mark.parametrize('solver', ['auto', 'lbfgs'])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('alpha', [0.001, 0.01, 0.1, 1.0])
def test_ridge_positive_regression_test(solver, fit_intercept, alpha):
    """Test that positive Ridge finds true positive coefficients."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    coef = np.array([1, -10])
    if fit_intercept:
        intercept = 20
        y = X.dot(coef) + intercept
    else:
        y = X.dot(coef)
    model = Ridge(alpha=alpha, positive=True, solver=solver, fit_intercept=fit_intercept)
    model.fit(X, y)
    assert np.all(model.coef_ >= 0)