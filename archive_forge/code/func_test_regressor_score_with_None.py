import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
@pytest.mark.parametrize('y,y_test', [([1, 1, 1, 2], [1.25] * 4), (np.array([[2, 2], [1, 1], [1, 1], [1, 1]]), [[1.25, 1.25]] * 4)])
def test_regressor_score_with_None(y, y_test):
    reg = DummyRegressor()
    reg.fit(None, y)
    assert reg.score(None, y_test) == 1.0