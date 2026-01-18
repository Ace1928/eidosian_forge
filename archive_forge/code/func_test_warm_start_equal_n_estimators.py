import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.metrics import check_scoring
@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
def test_warm_start_equal_n_estimators(GradientBoosting, X, y):
    gb_1 = GradientBoosting(max_depth=2, early_stopping=False)
    gb_1.fit(X, y)
    gb_2 = clone(gb_1)
    gb_2.set_params(max_iter=gb_1.max_iter, warm_start=True, n_iter_no_change=5)
    gb_2.fit(X, y)
    _assert_predictor_equal(gb_1, gb_2, X)