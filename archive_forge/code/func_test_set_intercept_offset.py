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
@pytest.mark.parametrize('klass, fit_params', [(SGDClassifier, {'intercept_init': np.zeros((3,))}), (SparseSGDClassifier, {'intercept_init': np.zeros((3,))}), (SGDOneClassSVM, {'offset_init': np.zeros((3,))}), (SparseSGDOneClassSVM, {'offset_init': np.zeros((3,))})])
def test_set_intercept_offset(klass, fit_params):
    """Check that `intercept_init` or `offset_init` is validated."""
    sgd_estimator = klass()
    with pytest.raises(ValueError, match='does not match dataset'):
        sgd_estimator.fit(X, Y, **fit_params)