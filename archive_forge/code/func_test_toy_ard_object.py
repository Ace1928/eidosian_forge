from math import log
import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_toy_ard_object():
    X = np.array([[1], [2], [3]])
    Y = np.array([1, 2, 3])
    clf = ARDRegression(compute_score=True)
    clf.fit(X, Y)
    test = [[1], [3], [4]]
    assert_array_almost_equal(clf.predict(test), [1, 3, 4], 2)