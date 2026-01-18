from math import log
import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_update_of_sigma_in_ard():
    X = np.array([[1, 0], [0, 0]])
    y = np.array([0, 0])
    clf = ARDRegression(max_iter=1)
    clf.fit(X, y)
    assert clf.sigma_.shape == (0, 0)
    clf.predict(X, return_std=True)