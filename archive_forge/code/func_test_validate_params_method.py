from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_validate_params_method():
    """Check that validate_params works with methods"""
    with pytest.raises(InvalidParameterError, match="The 'a' parameter of _Class._method must be"):
        _Class()._method('wrong')
    with pytest.warns(FutureWarning, match='Function _deprecated_method is deprecated'):
        with pytest.raises(InvalidParameterError, match="The 'a' parameter of _Class._deprecated_method must be"):
            _Class()._deprecated_method('wrong')