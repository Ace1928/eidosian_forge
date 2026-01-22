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
class TransAssert(BaseEstimator):

    def __init__(self, expected_type_transform):
        self.expected_type_transform = expected_type_transform

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, self.expected_type_transform)
        if isinstance(X, dataframe_lib.Series):
            X = X.to_frame()
        return X