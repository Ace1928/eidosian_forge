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
def test_sgd_clf(klass):
    for loss in ('hinge', 'squared_hinge', 'log_loss', 'modified_huber'):
        clf = klass(penalty='l2', alpha=0.01, fit_intercept=True, loss=loss, max_iter=10, shuffle=True)
        clf.fit(X, Y)
        assert_array_equal(clf.predict(T), true_result)