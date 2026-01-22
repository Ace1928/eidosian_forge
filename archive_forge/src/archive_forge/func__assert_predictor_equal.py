import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.metrics import check_scoring
def _assert_predictor_equal(gb_1, gb_2, X):
    """Assert that two HistGBM instances are identical."""
    for pred_ith_1, pred_ith_2 in zip(gb_1._predictors, gb_2._predictors):
        for predictor_1, predictor_2 in zip(pred_ith_1, pred_ith_2):
            assert_array_equal(predictor_1.nodes, predictor_2.nodes)
    assert_allclose(gb_1.predict(X), gb_2.predict(X))