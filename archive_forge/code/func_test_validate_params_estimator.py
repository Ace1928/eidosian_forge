from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_validate_params_estimator():
    """Check that validate_params works with Estimator instances"""
    est = _Estimator('wrong')
    with pytest.raises(InvalidParameterError, match="The 'a' parameter of _Estimator must be"):
        est.fit()