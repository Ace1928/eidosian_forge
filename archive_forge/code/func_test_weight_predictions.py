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
def test_weight_predictions():
    idata0 = from_dict(posterior_predictive={'a': np.random.normal(-1, 1, 1000)}, observed_data={'a': [1]})
    idata1 = from_dict(posterior_predictive={'a': np.random.normal(1, 1, 1000)}, observed_data={'a': [1]})
    new = weight_predictions([idata0, idata1])
    assert idata1.posterior_predictive.mean() > new.posterior_predictive.mean() > idata0.posterior_predictive.mean()
    assert 'posterior_predictive' in new
    assert 'observed_data' in new
    new = weight_predictions([idata0, idata1], weights=[0.5, 0.5])
    assert_almost_equal(new.posterior_predictive['a'].mean(), 0, decimal=1)
    new = weight_predictions([idata0, idata1], weights=[0.9, 0.1])
    assert_almost_equal(new.posterior_predictive['a'].mean(), -0.8, decimal=1)