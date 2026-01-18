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
def test_summary_nan(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.posterior['theta'].loc[{'school': 'Deerfield'}] = np.nan
    summary_xarray = summary(centered_eight)
    assert summary_xarray is not None
    assert summary_xarray.loc['theta[Deerfield]'].isnull().all()
    assert summary_xarray.loc[[ix for ix in summary_xarray.index if ix != 'theta[Deerfield]']].notnull().all().all()