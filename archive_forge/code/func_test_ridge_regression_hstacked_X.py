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
@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_ridge_regression_hstacked_X(solver, fit_intercept, ols_ridge_dataset, global_random_seed):
    """Test that Ridge converges for all solvers to correct solution on hstacked data.

    We work with a simple constructed data set with known solution.
    Fit on [X] with alpha is the same as fit on [X, X]/2 with alpha/2.
    For long X, [X, X] is a singular matrix.
    """
    X, y, _, coef = ols_ridge_dataset
    n_samples, n_features = X.shape
    alpha = 1.0
    model = Ridge(alpha=alpha / 2, fit_intercept=fit_intercept, solver=solver, tol=1e-15 if solver in ('sag', 'saga') else 1e-10, random_state=global_random_seed)
    X = X[:, :-1]
    X = 0.5 * np.concatenate((X, X), axis=1)
    assert np.linalg.matrix_rank(X) <= min(n_samples, n_features - 1)
    if fit_intercept:
        intercept = coef[-1]
    else:
        X = X - X.mean(axis=0)
        y = y - y.mean()
        intercept = 0
    model.fit(X, y)
    coef = coef[:-1]
    assert model.intercept_ == pytest.approx(intercept)
    assert_allclose(model.coef_, np.r_[coef, coef], atol=1e-08)