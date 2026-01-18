from unittest.mock import Mock
import numpy as np
import pytest
import scipy.stats as st
from ...data import dict_to_dataset, from_dict, load_arviz_data
from ...stats.density_utils import _circular_mean, _normalize_angle, _find_hdi_contours
from ...utils import (
from ..helpers import RandomVariableTestClass
@pytest.mark.parametrize('mean', [0, np.pi, 4 * np.pi, -2 * np.pi, -10 * np.pi])
def test_normalize_angle(mean):
    """Testing _normalize_angles() return values between expected bounds"""
    rvs = st.vonmises.rvs(loc=mean, kappa=1, size=1000)
    values = _normalize_angle(rvs, zero_centered=True)
    assert ((-np.pi <= values) & (values <= np.pi)).all()
    values = _normalize_angle(rvs, zero_centered=False)
    assert ((values >= 0) & (values <= 2 * np.pi)).all()