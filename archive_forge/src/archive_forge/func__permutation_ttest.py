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
def _permutation_ttest(a, b, permutations, axis=0, equal_var=True, nan_policy='propagate', random_state=None, alternative='two-sided'):
    """
    Calculates the T-test for the means of TWO INDEPENDENT samples of scores
    using permutation methods.

    This test is similar to `stats.ttest_ind`, except it doesn't rely on an
    approximate normality assumption since it uses a permutation test.
    This function is only called from ttest_ind when permutations is not None.

    Parameters
    ----------
    a, b : array_like
        The arrays must be broadcastable, except along the dimension
        corresponding to `axis` (the zeroth, by default).
    axis : int, optional
        The axis over which to operate on a and b.
    permutations : int, optional
        Number of permutations used to calculate p-value. If greater than or
        equal to the number of distinct permutations, perform an exact test.
    equal_var : bool, optional
        If False, an equal variance (Welch's) t-test is conducted.  Otherwise,
        an ordinary t-test is conducted.
    random_state : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.
        Pseudorandom number generator state used for generating random
        permutations.

    Returns
    -------
    statistic : float or array
        The calculated t-statistic.
    pvalue : float or array
        The p-value.

    """
    random_state = check_random_state(random_state)
    t_stat_observed = _calc_t_stat(a, b, equal_var, axis=axis)
    na = a.shape[axis]
    mat = _broadcast_concatenate((a, b), axis=axis)
    mat = np.moveaxis(mat, axis, -1)
    t_stat, permutations, n_max = _permutation_distribution_t(mat, permutations, size_a=na, equal_var=equal_var, random_state=random_state)
    compare = {'less': np.less_equal, 'greater': np.greater_equal, 'two-sided': lambda x, y: (x <= -np.abs(y)) | (x >= np.abs(y))}
    cmps = compare[alternative](t_stat, t_stat_observed)
    adjustment = 1 if n_max > permutations else 0
    pvalues = (cmps.sum(axis=0) + adjustment) / (permutations + adjustment)
    if nan_policy == 'propagate' and np.isnan(t_stat_observed).any():
        if np.ndim(pvalues) == 0:
            pvalues = np.float64(np.nan)
        else:
            pvalues[np.isnan(t_stat_observed)] = np.nan
    return (t_stat_observed, pvalues)