import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.metrics import check_scoring
@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
@pytest.mark.parametrize('scoring', (None, 'loss'))
def test_warm_start_early_stopping(GradientBoosting, X, y, scoring):
    n_iter_no_change = 5
    gb = GradientBoosting(n_iter_no_change=n_iter_no_change, max_iter=10000, early_stopping=True, random_state=42, warm_start=True, tol=0.001, scoring=scoring)
    gb.fit(X, y)
    n_iter_first_fit = gb.n_iter_
    gb.fit(X, y)
    n_iter_second_fit = gb.n_iter_
    assert 0 < n_iter_second_fit - n_iter_first_fit < n_iter_no_change