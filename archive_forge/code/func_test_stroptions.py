from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_stroptions():
    """Sanity check for the StrOptions constraint"""
    options = StrOptions({'a', 'b', 'c'}, deprecated={'c'})
    assert options.is_satisfied_by('a')
    assert options.is_satisfied_by('c')
    assert not options.is_satisfied_by('d')
    assert "'c' (deprecated)" in str(options)