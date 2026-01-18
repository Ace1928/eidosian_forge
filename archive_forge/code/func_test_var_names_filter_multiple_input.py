from unittest.mock import Mock
import numpy as np
import pytest
import scipy.stats as st
from ...data import dict_to_dataset, from_dict, load_arviz_data
from ...stats.density_utils import _circular_mean, _normalize_angle, _find_hdi_contours
from ...utils import (
from ..helpers import RandomVariableTestClass
@pytest.mark.parametrize('var_args', [(['ta'], ['beta1', 'beta2', 'theta'], 'like'), (['~beta'], ['phi', 'theta'], 'like'), (['beta[0-9]+'], ['beta1', 'beta2'], 'regex'), (['^p'], ['phi'], 'regex'), (['~^t'], ['beta1', 'beta2', 'phi'], 'regex')])
def test_var_names_filter_multiple_input(var_args):
    samples = np.random.randn(10)
    data1 = dict_to_dataset({'beta1': samples, 'beta2': samples, 'phi': samples})
    data2 = dict_to_dataset({'beta1': samples, 'beta2': samples, 'theta': samples})
    data = [data1, data2]
    var_names, expected, filter_vars = var_args
    assert _var_names(var_names, data, filter_vars) == expected