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
@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor])
def test_sgd_early_stopping_with_partial_fit(klass):
    """Check that we raise an error for `early_stopping` used with
    `partial_fit`.
    """
    err_msg = 'early_stopping should be False with partial_fit'
    with pytest.raises(ValueError, match=err_msg):
        klass(early_stopping=True).partial_fit(X, Y)