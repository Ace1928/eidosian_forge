from copy import deepcopy
import numpy as np
import pytest
from numpy.testing import (
from scipy.special import logsumexp
from scipy.stats import linregress, norm, halfcauchy
from xarray import DataArray, Dataset
from xarray_einstats.stats import XrContinuousRV
from ...data import concat, convert_to_inference_data, from_dict, load_arviz_data
from ...rcparams import rcParams
from ...stats import (
from ...stats.stats import _gpinv
from ...stats.stats_utils import get_log_likelihood
from ..helpers import check_multiple_attrs, multidim_models  # pylint: disable=unused-import
def test_priorsens_var_names(psens_data):
    result1 = psens(psens_data, component='prior', component_var_names=['mu', 'tau'], var_names=['mu', 'tau'])
    result2 = psens(psens_data, component='prior', var_names=['mu', 'tau'])
    for result in (result1, result2):
        assert 'theta' not in result
        assert 'mu' in result
        assert 'tau' in result
    assert not np.isclose(result1.mu, result2.mu)