import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def _check_behavior_2d_for_constant(clf):
    X = np.array([[0], [0], [0], [0]])
    y = np.array([[1, 0, 5, 4, 3], [2, 0, 1, 2, 5], [1, 0, 4, 5, 2], [1, 3, 3, 2, 0]])
    est = clone(clf)
    est.fit(X, y)
    y_pred = est.predict(X)
    assert y.shape == y_pred.shape