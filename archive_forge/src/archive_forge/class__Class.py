from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
class _Class:
    """A class to test the _InstancesOf constraint and the validation of methods."""

    @validate_params({'a': [Real]}, prefer_skip_nested_validation=True)
    def _method(self, a):
        """A validated method"""

    @deprecated()
    @validate_params({'a': [Real]}, prefer_skip_nested_validation=True)
    def _deprecated_method(self, a):
        """A deprecated validated method"""