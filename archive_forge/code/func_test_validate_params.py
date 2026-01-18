from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_validate_params():
    """Check that validate_params works no matter how the arguments are passed"""
    with pytest.raises(InvalidParameterError, match="The 'a' parameter of _func must be"):
        _func('wrong', c=1)
    with pytest.raises(InvalidParameterError, match="The 'b' parameter of _func must be"):
        _func(*[1, 'wrong'], c=1)
    with pytest.raises(InvalidParameterError, match="The 'c' parameter of _func must be"):
        _func(1, **{'c': 'wrong'})
    with pytest.raises(InvalidParameterError, match="The 'd' parameter of _func must be"):
        _func(1, c=1, d='wrong')
    with pytest.raises(InvalidParameterError, match="The 'b' parameter of _func must be"):
        _func(0, *['wrong', 2, 3], c=4, **{'e': 5})
    with pytest.raises(InvalidParameterError, match="The 'c' parameter of _func must be"):
        _func(0, *[1, 2, 3], c='four', **{'e': 5})