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
def test_psislw(centered_eight):
    pareto_k = loo(centered_eight, pointwise=True, reff=0.7)['pareto_k']
    log_likelihood = get_log_likelihood(centered_eight)
    log_likelihood = log_likelihood.stack(__sample__=('chain', 'draw'))
    assert_allclose(pareto_k, psislw(-log_likelihood, 0.7)[1])