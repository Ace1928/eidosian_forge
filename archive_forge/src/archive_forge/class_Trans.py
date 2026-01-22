import pickle
import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
class Trans(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if hasattr(X, 'to_frame'):
            return X.to_frame()
        if getattr(X, 'ndim', 2) == 1:
            return np.atleast_2d(X).T
        return X