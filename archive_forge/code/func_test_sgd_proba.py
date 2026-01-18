import pickle
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import datasets, linear_model, metrics
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import _sgd_fast as sgd_fast
from sklearn.linear_model import _stochastic_gradient
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, scale
from sklearn.svm import OneClassSVM
from sklearn.utils._testing import (
@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_sgd_proba(klass):
    clf = SGDClassifier(loss='hinge', alpha=0.01, max_iter=10, tol=None).fit(X, Y)
    assert not hasattr(clf, 'predict_proba')
    assert not hasattr(clf, 'predict_log_proba')
    for loss in ['log_loss', 'modified_huber']:
        clf = klass(loss=loss, alpha=0.01, max_iter=10)
        clf.fit(X, Y)
        p = clf.predict_proba([[3, 2]])
        assert p[0, 1] > 0.5
        p = clf.predict_proba([[-1, -1]])
        assert p[0, 1] < 0.5
        with np.errstate(divide='ignore'):
            p = clf.predict_log_proba([[3, 2]])
            assert p[0, 1] > p[0, 0]
            p = clf.predict_log_proba([[-1, -1]])
            assert p[0, 1] < p[0, 0]
    clf = klass(loss='log_loss', alpha=0.01, max_iter=10).fit(X2, Y2)
    d = clf.decision_function([[0.1, -0.1], [0.3, 0.2]])
    p = clf.predict_proba([[0.1, -0.1], [0.3, 0.2]])
    assert_array_equal(np.argmax(p, axis=1), np.argmax(d, axis=1))
    assert_almost_equal(p[0].sum(), 1)
    assert np.all(p[0] >= 0)
    p = clf.predict_proba([[-1, -1]])
    d = clf.decision_function([[-1, -1]])
    assert_array_equal(np.argsort(p[0]), np.argsort(d[0]))
    lp = clf.predict_log_proba([[3, 2]])
    p = clf.predict_proba([[3, 2]])
    assert_array_almost_equal(np.log(p), lp)
    lp = clf.predict_log_proba([[-1, -1]])
    p = clf.predict_proba([[-1, -1]])
    assert_array_almost_equal(np.log(p), lp)
    clf = klass(loss='modified_huber', alpha=0.01, max_iter=10)
    clf.fit(X2, Y2)
    d = clf.decision_function([[3, 2]])
    p = clf.predict_proba([[3, 2]])
    if klass != SparseSGDClassifier:
        assert np.argmax(d, axis=1) == np.argmax(p, axis=1)
    else:
        assert np.argmin(d, axis=1) == np.argmin(p, axis=1)
    x = X.mean(axis=0)
    d = clf.decision_function([x])
    if np.all(d < -1):
        p = clf.predict_proba([x])
        assert_array_almost_equal(p[0], [1 / 3.0] * 3)