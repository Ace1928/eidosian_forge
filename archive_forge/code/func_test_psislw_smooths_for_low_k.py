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
def test_psislw_smooths_for_low_k():
    rng = np.random.default_rng(44)
    x = rng.normal(size=100)
    x_smoothed, k = psislw(x.copy())
    assert k < 1 / 3
    assert not np.allclose(x - logsumexp(x), x_smoothed)