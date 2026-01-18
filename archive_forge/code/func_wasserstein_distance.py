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
def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
    """
    Compute the first Wasserstein distance between two 1D distributions.

    This distance is also known as the earth mover's distance, since it can be
    seen as the minimum amount of "work" required to transform :math:`u` into
    :math:`v`, where "work" is measured as the amount of distribution weight
    that must be moved, multiplied by the distance it has to be moved.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The first Wasserstein distance between the distributions :math:`u` and
    :math:`v` is:

    .. math::

        l_1 (u, v) = \\inf_{\\pi \\in \\Gamma (u, v)} \\int_{\\mathbb{R} \\times
        \\mathbb{R}} |x-y| \\mathrm{d} \\pi (x, y)

    where :math:`\\Gamma (u, v)` is the set of (probability) distributions on
    :math:`\\mathbb{R} \\times \\mathbb{R}` whose marginals are :math:`u` and
    :math:`v` on the first and second factors respectively.

    If :math:`U` and :math:`V` are the respective CDFs of :math:`u` and
    :math:`v`, this distance also equals to:

    .. math::

        l_1(u, v) = \\int_{-\\infty}^{+\\infty} |U-V|

    See [2]_ for a proof of the equivalence of both definitions.

    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] "Wasserstein metric", https://en.wikipedia.org/wiki/Wasserstein_metric
    .. [2] Ramdas, Garcia, Cuturi "On Wasserstein Two Sample Testing and Related
           Families of Nonparametric Tests" (2015). :arXiv:`1509.02237`.

    Examples
    --------
    >>> from scipy.stats import wasserstein_distance
    >>> wasserstein_distance([0, 1, 3], [5, 6, 8])
    5.0
    >>> wasserstein_distance([0, 1], [0, 1], [3, 1], [2, 2])
    0.25
    >>> wasserstein_distance([3.4, 3.9, 7.5, 7.8], [4.5, 1.4],
    ...                      [1.4, 0.9, 3.1, 7.2], [3.2, 3.5])
    4.0781331438047861

    """
    return _cdf_distance(1, u_values, v_values, u_weights, v_weights)