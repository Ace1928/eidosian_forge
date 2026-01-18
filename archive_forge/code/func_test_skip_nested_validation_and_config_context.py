from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
@pytest.mark.parametrize('skip_parameter_validation, prefer_skip_nested_validation, expected_skipped', [(True, True, True), (True, False, True), (False, True, True), (False, False, False)])
def test_skip_nested_validation_and_config_context(skip_parameter_validation, prefer_skip_nested_validation, expected_skipped):
    """Check interaction between global skip and local skip."""

    @validate_params({'a': [int]}, prefer_skip_nested_validation=prefer_skip_nested_validation)
    def g(a):
        return get_config()['skip_parameter_validation']
    with config_context(skip_parameter_validation=skip_parameter_validation):
        actual_skipped = g(1)
    assert actual_skipped == expected_skipped