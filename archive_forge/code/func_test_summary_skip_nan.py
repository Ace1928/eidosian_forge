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
def test_summary_skip_nan(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.posterior['theta'].loc[{'draw': slice(10), 'school': 'Deerfield'}] = np.nan
    summary_xarray = summary(centered_eight)
    theta_1 = summary_xarray.loc['theta[Deerfield]'].isnull()
    assert summary_xarray is not None
    assert ~theta_1[:4].all()
    assert theta_1[4:].all()