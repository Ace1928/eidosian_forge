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
@pytest.mark.parametrize('SGDEstimator', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor, SGDOneClassSVM, SparseSGDOneClassSVM])
def test_sgd_numerical_consistency(SGDEstimator):
    X_64 = X.astype(dtype=np.float64)
    Y_64 = np.array(Y, dtype=np.float64)
    X_32 = X.astype(dtype=np.float32)
    Y_32 = np.array(Y, dtype=np.float32)
    sgd_64 = SGDEstimator(max_iter=20)
    sgd_64.fit(X_64, Y_64)
    sgd_32 = SGDEstimator(max_iter=20)
    sgd_32.fit(X_32, Y_32)
    assert_allclose(sgd_64.coef_, sgd_32.coef_)