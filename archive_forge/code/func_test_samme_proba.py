import re
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble._weight_boosting import _samme_proba
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_samme_proba():
    probs = np.array([[1, 1e-06, 0], [0.19, 0.6, 0.2], [-999, 0.51, 0.5], [1e-06, 1, 1e-09]])
    probs /= np.abs(probs.sum(axis=1))[:, np.newaxis]

    class MockEstimator:

        def predict_proba(self, X):
            assert_array_equal(X.shape, probs.shape)
            return probs
    mock = MockEstimator()
    samme_proba = _samme_proba(mock, 3, np.ones_like(probs))
    assert_array_equal(samme_proba.shape, probs.shape)
    assert np.isfinite(samme_proba).all()
    assert_array_equal(np.argmin(samme_proba, axis=1), [2, 0, 0, 2])
    assert_array_equal(np.argmax(samme_proba, axis=1), [0, 1, 1, 1])