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
def test_sgd_error_on_zero_validation_weight():
    X, Y = (iris.data, iris.target)
    sample_weight = np.zeros_like(Y)
    validation_fraction = 0.4
    clf = linear_model.SGDClassifier(early_stopping=True, validation_fraction=validation_fraction, random_state=0)
    error_message = 'The sample weights for validation set are all zero, consider using a different random state.'
    with pytest.raises(ValueError, match=error_message):
        clf.fit(X, Y, sample_weight=sample_weight)