import os
import re
import sys
import tempfile
import warnings
from functools import partial
from io import StringIO
from time import sleep
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.model_selection import (
from sklearn.model_selection._validation import (
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.model_selection.tests.test_search import FailingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import shuffle
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
class MockClassifier:
    """Dummy classifier to test the cross-validation"""

    def __init__(self, a=0, allow_nd=False):
        self.a = a
        self.allow_nd = allow_nd

    def fit(self, X, Y=None, sample_weight=None, class_prior=None, sparse_sample_weight=None, sparse_param=None, dummy_int=None, dummy_str=None, dummy_obj=None, callback=None):
        """The dummy arguments are to test that this fit function can
        accept non-array arguments through cross-validation, such as:
            - int
            - str (this is actually array-like)
            - object
            - function
        """
        self.dummy_int = dummy_int
        self.dummy_str = dummy_str
        self.dummy_obj = dummy_obj
        if callback is not None:
            callback(self)
        if self.allow_nd:
            X = X.reshape(len(X), -1)
        if X.ndim >= 3 and (not self.allow_nd):
            raise ValueError('X cannot be d')
        if sample_weight is not None:
            assert sample_weight.shape[0] == X.shape[0], 'MockClassifier extra fit_param sample_weight.shape[0] is {0}, should be {1}'.format(sample_weight.shape[0], X.shape[0])
        if class_prior is not None:
            assert class_prior.shape[0] == len(np.unique(y)), 'MockClassifier extra fit_param class_prior.shape[0] is {0}, should be {1}'.format(class_prior.shape[0], len(np.unique(y)))
        if sparse_sample_weight is not None:
            fmt = 'MockClassifier extra fit_param sparse_sample_weight.shape[0] is {0}, should be {1}'
            assert sparse_sample_weight.shape[0] == X.shape[0], fmt.format(sparse_sample_weight.shape[0], X.shape[0])
        if sparse_param is not None:
            fmt = 'MockClassifier extra fit_param sparse_param.shape is ({0}, {1}), should be ({2}, {3})'
            assert sparse_param.shape == P.shape, fmt.format(sparse_param.shape[0], sparse_param.shape[1], P.shape[0], P.shape[1])
        return self

    def predict(self, T):
        if self.allow_nd:
            T = T.reshape(len(T), -1)
        return T[:, 0]

    def predict_proba(self, T):
        return T

    def score(self, X=None, Y=None):
        return 1.0 / (1 + np.abs(self.a))

    def get_params(self, deep=False):
        return {'a': self.a, 'allow_nd': self.allow_nd}