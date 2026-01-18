from unittest.mock import Mock
import numpy as np
import pytest
import scipy.stats as st
from ...data import dict_to_dataset, from_dict, load_arviz_data
from ...stats.density_utils import _circular_mean, _normalize_angle, _find_hdi_contours
from ...utils import (
from ..helpers import RandomVariableTestClass
def test_nonstring_var_names():
    """Check that non-string variables are preserved"""
    mu = RandomVariableTestClass('mu')
    samples = np.random.randn(10)
    data = dict_to_dataset({mu: samples})
    assert _var_names([mu], data) == [mu]