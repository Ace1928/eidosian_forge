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
def tiecorrect(rankvals):
    """Tie correction factor for Mann-Whitney U and Kruskal-Wallis H tests.

    Parameters
    ----------
    rankvals : array_like
        A 1-D sequence of ranks.  Typically this will be the array
        returned by `~scipy.stats.rankdata`.

    Returns
    -------
    factor : float
        Correction factor for U or H.

    See Also
    --------
    rankdata : Assign ranks to the data
    mannwhitneyu : Mann-Whitney rank test
    kruskal : Kruskal-Wallis H test

    References
    ----------
    .. [1] Siegel, S. (1956) Nonparametric Statistics for the Behavioral
           Sciences.  New York: McGraw-Hill.

    Examples
    --------
    >>> from scipy.stats import tiecorrect, rankdata
    >>> tiecorrect([1, 2.5, 2.5, 4])
    0.9
    >>> ranks = rankdata([1, 3, 2, 4, 5, 7, 2, 8, 4])
    >>> ranks
    array([ 1. ,  4. ,  2.5,  5.5,  7. ,  8. ,  2.5,  9. ,  5.5])
    >>> tiecorrect(ranks)
    0.9833333333333333

    """
    arr = np.sort(rankvals)
    idx = np.nonzero(np.r_[True, arr[1:] != arr[:-1], True])[0]
    cnt = np.diff(idx).astype(np.float64)
    size = np.float64(arr.size)
    return 1.0 if size < 2 else 1.0 - (cnt ** 3 - cnt).sum() / (size ** 3 - size)