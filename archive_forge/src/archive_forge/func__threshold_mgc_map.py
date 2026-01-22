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
def _threshold_mgc_map(stat_mgc_map, samp_size):
    """
    Finds a connected region of significance in the MGC-map by thresholding.

    Parameters
    ----------
    stat_mgc_map : ndarray
        All local correlations within `[-1,1]`.
    samp_size : int
        The sample size of original data.

    Returns
    -------
    sig_connect : ndarray
        A binary matrix with 1's indicating the significant region.

    """
    m, n = stat_mgc_map.shape
    per_sig = 1 - 0.02 / samp_size
    threshold = samp_size * (samp_size - 3) / 4 - 1 / 2
    threshold = distributions.beta.ppf(per_sig, threshold, threshold) * 2 - 1
    threshold = max(threshold, stat_mgc_map[m - 1][n - 1])
    sig_connect = stat_mgc_map > threshold
    if np.sum(sig_connect) > 0:
        sig_connect, _ = _measurements.label(sig_connect)
        _, label_counts = np.unique(sig_connect, return_counts=True)
        max_label = np.argmax(label_counts[1:]) + 1
        sig_connect = sig_connect == max_label
    else:
        sig_connect = np.array([[False]])
    return sig_connect