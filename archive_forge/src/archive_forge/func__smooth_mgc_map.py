import warnings
import math
from math import gcd
from collections import namedtuple
import numpy as np
from numpy import array, asarray, ma
from scipy.spatial.distance import cdist
from scipy.ndimage import _measurements
from scipy._lib._util import (check_random_state, MapWrapper, _get_nan,
import scipy.special as special
from scipy import linalg
from . import distributions
from . import _mstats_basic as mstats_basic
from ._stats_mstats_common import (_find_repeats, linregress, theilslopes,
from ._stats import (_kendall_dis, _toint64, _weightedrankedtau,
from dataclasses import dataclass, field
from ._hypotests import _all_partitions
from ._stats_pythran import _compute_outer_prob_inside_method
from ._resampling import (MonteCarloMethod, PermutationMethod, BootstrapMethod,
from ._axis_nan_policy import (_axis_nan_policy_factory,
from ._binomtest import _binary_search_for_binom_tst as _binary_search
from scipy._lib._bunch import _make_tuple_bunch
from scipy import stats
from scipy.optimize import root_scalar
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy._lib._util import normalize_axis_index
from scipy._lib._util import float_factorial  # noqa: F401
from scipy.stats._mstats_basic import (  # noqa: F401
def _smooth_mgc_map(sig_connect, stat_mgc_map):
    """Finds the smoothed maximal within the significant region R.

    If area of R is too small it returns the last local correlation. Otherwise,
    returns the maximum within significant_connected_region.

    Parameters
    ----------
    sig_connect : ndarray
        A binary matrix with 1's indicating the significant region.
    stat_mgc_map : ndarray
        All local correlations within `[-1, 1]`.

    Returns
    -------
    stat : float
        The sample MGC statistic within `[-1, 1]`.
    opt_scale: (float, float)
        The estimated optimal scale as an `(x, y)` pair.

    """
    m, n = stat_mgc_map.shape
    stat = stat_mgc_map[m - 1][n - 1]
    opt_scale = [m, n]
    if np.linalg.norm(sig_connect) != 0:
        if np.sum(sig_connect) >= np.ceil(0.02 * max(m, n)) * min(m, n):
            max_corr = max(stat_mgc_map[sig_connect])
            max_corr_index = np.where((stat_mgc_map >= max_corr) & sig_connect)
            if max_corr >= stat:
                stat = max_corr
                k, l = max_corr_index
                one_d_indices = k * n + l
                k = np.max(one_d_indices) // n
                l = np.max(one_d_indices) % n
                opt_scale = [k + 1, l + 1]
    return (stat, opt_scale)