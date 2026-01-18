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
@pytest.mark.parametrize('scale', ['log', 'negative_log', 'deviance'])
def test_waic_print(centered_eight, scale):
    waic_data = repr(waic(centered_eight, scale=scale))
    waic_pointwise = repr(waic(centered_eight, scale=scale, pointwise=True))
    assert waic_data is not None
    assert waic_pointwise is not None
    assert waic_data == waic_pointwise