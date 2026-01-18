from unittest.mock import Mock
import numpy as np
import pytest
import scipy.stats as st
from ...data import dict_to_dataset, from_dict, load_arviz_data
from ...stats.density_utils import _circular_mean, _normalize_angle, _find_hdi_contours
from ...utils import (
from ..helpers import RandomVariableTestClass
@pytest.mark.parametrize('var_args', [(['alpha', 'beta'], ['alpha', 'beta1', 'beta2'], 'like'), (['~beta'], ['alpha', 'p1', 'p2', 'phi', 'theta', 'theta_t'], 'like'), (['theta'], ['theta', 'theta_t'], 'like'), (['~theta'], ['alpha', 'beta1', 'beta2', 'p1', 'p2', 'phi'], 'like'), (['p'], ['alpha', 'p1', 'p2', 'phi'], 'like'), (['~p'], ['beta1', 'beta2', 'theta', 'theta_t'], 'like'), (['^bet'], ['beta1', 'beta2'], 'regex'), (['^p'], ['p1', 'p2', 'phi'], 'regex'), (['~^p'], ['alpha', 'beta1', 'beta2', 'theta', 'theta_t'], 'regex'), (['p[0-9]+'], ['p1', 'p2'], 'regex'), (['~p[0-9]+'], ['alpha', 'beta1', 'beta2', 'phi', 'theta', 'theta_t'], 'regex')])
def test_var_names_filter(var_args):
    """Test var_names filter with partial naming or regular expressions."""
    samples = np.random.randn(10)
    data = dict_to_dataset({'alpha': samples, 'beta1': samples, 'beta2': samples, 'p1': samples, 'p2': samples, 'phi': samples, 'theta': samples, 'theta_t': samples})
    var_names, expected, filter_vars = var_args
    assert _var_names(var_names, data, filter_vars) == expected