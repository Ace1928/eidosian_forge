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
class _SparseSGDOneClassSVM(linear_model.SGDOneClassSVM):

    def fit(self, X, *args, **kw):
        X = sp.csr_matrix(X)
        return linear_model.SGDOneClassSVM.fit(self, X, *args, **kw)

    def partial_fit(self, X, *args, **kw):
        X = sp.csr_matrix(X)
        return linear_model.SGDOneClassSVM.partial_fit(self, X, *args, **kw)

    def decision_function(self, X, *args, **kw):
        X = sp.csr_matrix(X)
        return linear_model.SGDOneClassSVM.decision_function(self, X, *args, **kw)