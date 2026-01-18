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
def test_summary_labels():
    coords1 = list('abcd')
    coords2 = np.arange(1, 6)
    data = from_dict({'a': np.random.randn(4, 100, 4, 5)}, coords={'dim1': coords1, 'dim2': coords2}, dims={'a': ['dim1', 'dim2']})
    az_summary = summary(data, fmt='wide')
    assert az_summary is not None
    column_order = []
    for coord1 in coords1:
        for coord2 in coords2:
            column_order.append(f'a[{coord1}, {coord2}]')
    for col1, col2 in zip(list(az_summary.index), column_order):
        assert col1 == col2