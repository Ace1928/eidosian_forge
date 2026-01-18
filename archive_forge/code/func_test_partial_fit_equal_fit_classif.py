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
@pytest.mark.parametrize('lr', ['constant', 'optimal', 'invscaling', 'adaptive'])
def test_partial_fit_equal_fit_classif(klass, lr):
    for X_, Y_, T_ in ((X, Y, T), (X2, Y2, T2)):
        clf = klass(alpha=0.01, eta0=0.01, max_iter=2, learning_rate=lr, shuffle=False)
        clf.fit(X_, Y_)
        y_pred = clf.decision_function(T_)
        t = clf.t_
        classes = np.unique(Y_)
        clf = klass(alpha=0.01, eta0=0.01, learning_rate=lr, shuffle=False)
        for i in range(2):
            clf.partial_fit(X_, Y_, classes=classes)
        y_pred2 = clf.decision_function(T_)
        assert clf.t_ == t
        assert_array_almost_equal(y_pred, y_pred2, decimal=2)