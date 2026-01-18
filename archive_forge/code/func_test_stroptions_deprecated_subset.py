from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_stroptions_deprecated_subset():
    """Check that the deprecated parameter must be a subset of options."""
    with pytest.raises(ValueError, match='deprecated options must be a subset'):
        StrOptions({'a', 'b', 'c'}, deprecated={'a', 'd'})