from unittest.mock import Mock
import numpy as np
import pytest
import scipy.stats as st
from ...data import dict_to_dataset, from_dict, load_arviz_data
from ...stats.density_utils import _circular_mean, _normalize_angle, _find_hdi_contours
from ...utils import (
from ..helpers import RandomVariableTestClass
def test_var_names_key_error(data):
    with pytest.raises(KeyError, match='bad_var_name'):
        _var_names(('theta', 'tau', 'bad_var_name'), data)