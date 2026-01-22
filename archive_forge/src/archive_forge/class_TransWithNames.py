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
class TransWithNames(Trans):

    def __init__(self, feature_names_out=None):
        self.feature_names_out = feature_names_out

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out is not None:
            return np.asarray(self.feature_names_out, dtype=object)
        return input_features