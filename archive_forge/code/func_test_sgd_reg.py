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
@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
def test_sgd_reg(klass):
    clf = klass(alpha=0.1, max_iter=2, fit_intercept=False)
    clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    assert clf.coef_[0] == clf.coef_[1]