from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_skip_param_validation():
    """Check that param validation can be skipped using config_context."""

    @validate_params({'a': [int]}, prefer_skip_nested_validation=True)
    def f(a):
        pass
    with pytest.raises(InvalidParameterError, match="The 'a' parameter"):
        f(a='1')
    with config_context(skip_parameter_validation=True):
        f(a='1')