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
def _mgc_stat(distx, disty):
    """Helper function that calculates the MGC stat. See above for use.

    Parameters
    ----------
    distx, disty : ndarray
        `distx` and `disty` have shapes `(n, p)` and `(n, q)` or
        `(n, n)` and `(n, n)`
        if distance matrices.

    Returns
    -------
    stat : float
        The sample MGC test statistic within `[-1, 1]`.
    stat_dict : dict
        Contains additional useful additional returns containing the following
        keys:

            - stat_mgc_map : ndarray
                MGC-map of the statistics.
            - opt_scale : (float, float)
                The estimated optimal scale as a `(x, y)` pair.

    """
    stat_mgc_map = _local_correlations(distx, disty, global_corr='mgc')
    n, m = stat_mgc_map.shape
    if m == 1 or n == 1:
        stat = stat_mgc_map[m - 1][n - 1]
        opt_scale = m * n
    else:
        samp_size = len(distx) - 1
        sig_connect = _threshold_mgc_map(stat_mgc_map, samp_size)
        stat, opt_scale = _smooth_mgc_map(sig_connect, stat_mgc_map)
    stat_dict = {'stat_mgc_map': stat_mgc_map, 'opt_scale': opt_scale}
    return (stat, stat_dict)