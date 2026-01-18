import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_quantile_invalid():
    X = [[0]] * 5
    y = [0] * 5
    est = DummyRegressor(strategy='quantile', quantile=None)
    err_msg = "When using `strategy='quantile', you have to specify the desired quantile"
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)