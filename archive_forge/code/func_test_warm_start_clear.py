import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.metrics import check_scoring
@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
def test_warm_start_clear(GradientBoosting, X, y):
    gb_1 = GradientBoosting(n_iter_no_change=5, random_state=42)
    gb_1.fit(X, y)
    gb_2 = GradientBoosting(n_iter_no_change=5, random_state=42, warm_start=True)
    gb_2.fit(X, y)
    gb_2.set_params(warm_start=False)
    gb_2.fit(X, y)
    assert_allclose(gb_1.train_score_, gb_2.train_score_)
    assert_allclose(gb_1.validation_score_, gb_2.validation_score_)
    _assert_predictor_equal(gb_1, gb_2, X)