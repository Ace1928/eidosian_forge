from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
class SimpleEstimator(BaseEstimator):

    def fit(self, X, y=None):
        self.X_ = X
        self.n_features_ = X.shape[0]
        return self