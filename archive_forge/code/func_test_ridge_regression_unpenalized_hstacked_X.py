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
def test_ridge_regression_unpenalized_hstacked_X(solver, fit_intercept, ols_ridge_dataset, global_random_seed):
    """Test that unpenalized Ridge = OLS converges for all solvers to correct solution.

    We work with a simple constructed data set with known solution.
    OLS fit on [X] is the same as fit on [X, X]/2.
    For long X, [X, X] is a singular matrix and we check against the minimum norm
    solution:
        min ||w||_2 subject to min ||X w - y||_2
    """
    X, y, coef, _ = ols_ridge_dataset
    n_samples, n_features = X.shape
    alpha = 0
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver, tol=1e-15 if solver in ('sag', 'saga') else 1e-10, random_state=global_random_seed)
    if fit_intercept:
        X = X[:, :-1]
        intercept = coef[-1]
        coef = coef[:-1]
    else:
        intercept = 0
    X = 0.5 * np.concatenate((X, X), axis=1)
    assert np.linalg.matrix_rank(X) <= min(n_samples, n_features)
    model.fit(X, y)
    if n_samples > n_features or not fit_intercept:
        assert model.intercept_ == pytest.approx(intercept)
        if solver == 'cholesky':
            pytest.skip()
        assert_allclose(model.coef_, np.r_[coef, coef])
    else:
        assert_allclose(model.predict(X), y)
        assert np.linalg.norm(np.r_[model.intercept_, model.coef_]) > np.linalg.norm(np.r_[intercept, coef, coef])
        pytest.xfail(reason='Ridge does not provide the minimum norm solution.')
        assert model.intercept_ == pytest.approx(intercept)
        assert_allclose(model.coef_, np.r_[coef, coef])