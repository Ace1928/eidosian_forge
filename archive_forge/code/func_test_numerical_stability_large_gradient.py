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
def test_numerical_stability_large_gradient():
    model = SGDClassifier(loss='squared_hinge', max_iter=10, shuffle=True, penalty='elasticnet', l1_ratio=0.3, alpha=0.01, eta0=0.001, random_state=0, tol=None)
    with np.errstate(all='raise'):
        model.fit(iris.data, iris.target)
    assert np.isfinite(model.coef_).all()