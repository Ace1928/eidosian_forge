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
def test_underflow_or_overlow():
    with np.errstate(all='raise'):
        rng = np.random.RandomState(0)
        n_samples = 100
        n_features = 10
        X = rng.normal(size=(n_samples, n_features))
        X[:, :2] *= 1e+300
        assert np.isfinite(X).all()
        X_scaled = MinMaxScaler().fit_transform(X)
        assert np.isfinite(X_scaled).all()
        ground_truth = rng.normal(size=n_features)
        y = (np.dot(X_scaled, ground_truth) > 0.0).astype(np.int32)
        assert_array_equal(np.unique(y), [0, 1])
        model = SGDClassifier(alpha=0.1, loss='squared_hinge', max_iter=500)
        model.fit(X_scaled, y)
        assert np.isfinite(model.coef_).all()
        msg_regxp = 'Floating-point under-/overflow occurred at epoch #.* Scaling input data with StandardScaler or MinMaxScaler might help.'
        with pytest.raises(ValueError, match=msg_regxp):
            model.fit(X, y)