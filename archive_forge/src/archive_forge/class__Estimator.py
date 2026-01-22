from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
class _Estimator(BaseEstimator):
    """An estimator to test the validation of estimator parameters."""
    _parameter_constraints: dict = {'a': [Real]}

    def __init__(self, a):
        self.a = a

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X=None, y=None):
        pass