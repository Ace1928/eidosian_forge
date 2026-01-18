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
@pytest.mark.parametrize('params', (('mean', 'all', METRICS_NAMES[:9]), ('mean', 'stats', METRICS_NAMES[:4]), ('mean', 'diagnostics', METRICS_NAMES[4:9]), ('median', 'all', METRICS_NAMES[9:17]), ('median', 'stats', METRICS_NAMES[9:13]), ('median', 'diagnostics', METRICS_NAMES[13:17])))
def test_summary_focus_kind(centered_eight, params):
    stat_focus, kind, metrics_names_ = params
    summary_df = summary(centered_eight, stat_focus=stat_focus, kind=kind)
    assert_array_equal(summary_df.columns, metrics_names_)