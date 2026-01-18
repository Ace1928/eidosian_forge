import re
import warnings
from unittest.mock import Mock
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.datasets import make_friedman1
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import (
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.utils._testing import (
def test_prefit():
    clf = SGDClassifier(alpha=0.1, max_iter=10, shuffle=True, random_state=0, tol=None)
    model = SelectFromModel(clf)
    model.fit(data, y)
    X_transform = model.transform(data)
    clf.fit(data, y)
    model = SelectFromModel(clf, prefit=True)
    assert_array_almost_equal(model.transform(data), X_transform)
    model.fit(data, y)
    assert model.estimator_ is not clf
    model = SelectFromModel(clf, prefit=False)
    model.fit(data, y)
    assert_array_almost_equal(model.transform(data), X_transform)
    clf = SGDClassifier(alpha=0.1, max_iter=10, shuffle=True, random_state=0, tol=None)
    model = SelectFromModel(clf, prefit=True)
    err_msg = 'When `prefit=True`, `estimator` is expected to be a fitted estimator.'
    with pytest.raises(NotFittedError, match=err_msg):
        model.fit(data, y)
    with pytest.raises(NotFittedError, match=err_msg):
        model.partial_fit(data, y)
    with pytest.raises(NotFittedError, match=err_msg):
        model.transform(data)
    clf = SGDClassifier(alpha=0.1, max_iter=10, shuffle=True, tol=None).fit(data, y)
    model = SelectFromModel(clf, prefit=True)
    model.fit(data, y)
    assert_allclose(model.estimator_.coef_, clf.coef_)
    model.partial_fit(data, y)
    assert_allclose(model.estimator_.coef_, clf.coef_)