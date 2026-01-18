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
def test_sgd_oneclass():
    X_train = np.array([[-2, -1], [-1, -1], [1, 1]])
    X_test = np.array([[0.5, -2], [2, 2]])
    clf = SGDOneClassSVM(nu=0.5, eta0=1, learning_rate='constant', shuffle=False, max_iter=1)
    clf.fit(X_train)
    assert_allclose(clf.coef_, np.array([-0.125, 0.4375]))
    assert clf.offset_[0] == -0.5
    scores = clf.score_samples(X_test)
    assert_allclose(scores, np.array([-0.9375, 0.625]))
    dec = clf.score_samples(X_test) - clf.offset_
    assert_allclose(clf.decision_function(X_test), dec)
    pred = clf.predict(X_test)
    assert_array_equal(pred, np.array([-1, 1]))