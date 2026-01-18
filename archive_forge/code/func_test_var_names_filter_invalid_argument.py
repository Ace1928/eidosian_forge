from unittest.mock import Mock
import numpy as np
import pytest
import scipy.stats as st
from ...data import dict_to_dataset, from_dict, load_arviz_data
from ...stats.density_utils import _circular_mean, _normalize_angle, _find_hdi_contours
from ...utils import (
from ..helpers import RandomVariableTestClass
def test_var_names_filter_invalid_argument():
    """Check invalid argument raises."""
    samples = np.random.randn(10)
    data = dict_to_dataset({'alpha': samples})
    msg = "^\\'filter_vars\\' can only be None, \\'like\\', or \\'regex\\', got: 'foo'$"
    with pytest.raises(ValueError, match=msg):
        assert _var_names(['alpha'], data, filter_vars='foo')