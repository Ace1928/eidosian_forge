from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_validate_params_missing_params():
    """Check that no error is raised when there are parameters without
    constraints
    """

    @validate_params({'a': [int]}, prefer_skip_nested_validation=True)
    def func(a, b):
        pass
    func(1, 2)