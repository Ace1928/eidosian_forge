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
def weightedtau(x, y, rank=True, weigher=None, additive=True):
    """Compute a weighted version of Kendall's :math:`\\tau`.

    The weighted :math:`\\tau` is a weighted version of Kendall's
    :math:`\\tau` in which exchanges of high weight are more influential than
    exchanges of low weight. The default parameters compute the additive
    hyperbolic version of the index, :math:`\\tau_\\mathrm h`, which has
    been shown to provide the best balance between important and
    unimportant elements [1]_.

    The weighting is defined by means of a rank array, which assigns a
    nonnegative rank to each element (higher importance ranks being
    associated with smaller values, e.g., 0 is the highest possible rank),
    and a weigher function, which assigns a weight based on the rank to
    each element. The weight of an exchange is then the sum or the product
    of the weights of the ranks of the exchanged elements. The default
    parameters compute :math:`\\tau_\\mathrm h`: an exchange between
    elements with rank :math:`r` and :math:`s` (starting from zero) has
    weight :math:`1/(r+1) + 1/(s+1)`.

    Specifying a rank array is meaningful only if you have in mind an
    external criterion of importance. If, as it usually happens, you do
    not have in mind a specific rank, the weighted :math:`\\tau` is
    defined by averaging the values obtained using the decreasing
    lexicographical rank by (`x`, `y`) and by (`y`, `x`). This is the
    behavior with default parameters. Note that the convention used
    here for ranking (lower values imply higher importance) is opposite
    to that used by other SciPy statistical functions.

    Parameters
    ----------
    x, y : array_like
        Arrays of scores, of the same shape. If arrays are not 1-D, they will
        be flattened to 1-D.
    rank : array_like of ints or bool, optional
        A nonnegative rank assigned to each element. If it is None, the
        decreasing lexicographical rank by (`x`, `y`) will be used: elements of
        higher rank will be those with larger `x`-values, using `y`-values to
        break ties (in particular, swapping `x` and `y` will give a different
        result). If it is False, the element indices will be used
        directly as ranks. The default is True, in which case this
        function returns the average of the values obtained using the
        decreasing lexicographical rank by (`x`, `y`) and by (`y`, `x`).
    weigher : callable, optional
        The weigher function. Must map nonnegative integers (zero
        representing the most important element) to a nonnegative weight.
        The default, None, provides hyperbolic weighing, that is,
        rank :math:`r` is mapped to weight :math:`1/(r+1)`.
    additive : bool, optional
        If True, the weight of an exchange is computed by adding the
        weights of the ranks of the exchanged elements; otherwise, the weights
        are multiplied. The default is True.

    Returns
    -------
    res: SignificanceResult
        An object containing attributes:

        statistic : float
           The weighted :math:`\\tau` correlation index.
        pvalue : float
           Presently ``np.nan``, as the null distribution of the statistic is
           unknown (even in the additive hyperbolic case).

    See Also
    --------
    kendalltau : Calculates Kendall's tau.
    spearmanr : Calculates a Spearman rank-order correlation coefficient.
    theilslopes : Computes the Theil-Sen estimator for a set of points (x, y).

    Notes
    -----
    This function uses an :math:`O(n \\log n)`, mergesort-based algorithm
    [1]_ that is a weighted extension of Knight's algorithm for Kendall's
    :math:`\\tau` [2]_. It can compute Shieh's weighted :math:`\\tau` [3]_
    between rankings without ties (i.e., permutations) by setting
    `additive` and `rank` to False, as the definition given in [1]_ is a
    generalization of Shieh's.

    NaNs are considered the smallest possible score.

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Sebastiano Vigna, "A weighted correlation index for rankings with
           ties", Proceedings of the 24th international conference on World
           Wide Web, pp. 1166-1176, ACM, 2015.
    .. [2] W.R. Knight, "A Computer Method for Calculating Kendall's Tau with
           Ungrouped Data", Journal of the American Statistical Association,
           Vol. 61, No. 314, Part 1, pp. 436-439, 1966.
    .. [3] Grace S. Shieh. "A weighted Kendall's tau statistic", Statistics &
           Probability Letters, Vol. 39, No. 1, pp. 17-24, 1998.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> x = [12, 2, 1, 12, 2]
    >>> y = [1, 4, 7, 1, 0]
    >>> res = stats.weightedtau(x, y)
    >>> res.statistic
    -0.56694968153682723
    >>> res.pvalue
    nan
    >>> res = stats.weightedtau(x, y, additive=False)
    >>> res.statistic
    -0.62205716951801038

    NaNs are considered the smallest possible score:

    >>> x = [12, 2, 1, 12, 2]
    >>> y = [1, 4, 7, 1, np.nan]
    >>> res = stats.weightedtau(x, y)
    >>> res.statistic
    -0.56694968153682723

    This is exactly Kendall's tau:

    >>> x = [12, 2, 1, 12, 2]
    >>> y = [1, 4, 7, 1, 0]
    >>> res = stats.weightedtau(x, y, weigher=lambda x: 1)
    >>> res.statistic
    -0.47140452079103173

    >>> x = [12, 2, 1, 12, 2]
    >>> y = [1, 4, 7, 1, 0]
    >>> stats.weightedtau(x, y, rank=None)
    SignificanceResult(statistic=-0.4157652301037516, pvalue=nan)
    >>> stats.weightedtau(y, x, rank=None)
    SignificanceResult(statistic=-0.7181341329699028, pvalue=nan)

    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size:
        raise ValueError(f'All inputs to `weightedtau` must be of the same size, found x-size {x.size} and y-size {y.size}')
    if not x.size:
        res = SignificanceResult(np.nan, np.nan)
        res.correlation = np.nan
        return res
    if np.isnan(np.sum(x)):
        x = _toint64(x)
    if np.isnan(np.sum(y)):
        y = _toint64(y)
    if x.dtype != y.dtype:
        if x.dtype != np.int64:
            x = _toint64(x)
        if y.dtype != np.int64:
            y = _toint64(y)
    elif x.dtype not in (np.int32, np.int64, np.float32, np.float64):
        x = _toint64(x)
        y = _toint64(y)
    if rank is True:
        tau = (_weightedrankedtau(x, y, None, weigher, additive) + _weightedrankedtau(y, x, None, weigher, additive)) / 2
        res = SignificanceResult(tau, np.nan)
        res.correlation = tau
        return res
    if rank is False:
        rank = np.arange(x.size, dtype=np.intp)
    elif rank is not None:
        rank = np.asarray(rank).ravel()
        if rank.size != x.size:
            raise ValueError(f'All inputs to `weightedtau` must be of the same size, found x-size {x.size} and rank-size {rank.size}')
    tau = _weightedrankedtau(x, y, rank, weigher, additive)
    res = SignificanceResult(tau, np.nan)
    res.correlation = tau
    return res