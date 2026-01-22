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
def _histogram(a, numbins=10, defaultlimits=None, weights=None, printextras=False):
    """Create a histogram.

    Separate the range into several bins and return the number of instances
    in each bin.

    Parameters
    ----------
    a : array_like
        Array of scores which will be put into bins.
    numbins : int, optional
        The number of bins to use for the histogram. Default is 10.
    defaultlimits : tuple (lower, upper), optional
        The lower and upper values for the range of the histogram.
        If no value is given, a range slightly larger than the range of the
        values in a is used. Specifically ``(a.min() - s, a.max() + s)``,
        where ``s = (1/2)(a.max() - a.min()) / (numbins - 1)``.
    weights : array_like, optional
        The weights for each value in `a`. Default is None, which gives each
        value a weight of 1.0
    printextras : bool, optional
        If True, if there are extra points (i.e. the points that fall outside
        the bin limits) a warning is raised saying how many of those points
        there are.  Default is False.

    Returns
    -------
    count : ndarray
        Number of points (or sum of weights) in each bin.
    lowerlimit : float
        Lowest value of histogram, the lower limit of the first bin.
    binsize : float
        The size of the bins (all bins have the same size).
    extrapoints : int
        The number of points outside the range of the histogram.

    See Also
    --------
    numpy.histogram

    Notes
    -----
    This histogram is based on numpy's histogram but has a larger range by
    default if default limits is not set.

    """
    a = np.ravel(a)
    if defaultlimits is None:
        if a.size == 0:
            defaultlimits = (0, 1)
        else:
            data_min = a.min()
            data_max = a.max()
            s = (data_max - data_min) / (2.0 * (numbins - 1.0))
            defaultlimits = (data_min - s, data_max + s)
    hist, bin_edges = np.histogram(a, bins=numbins, range=defaultlimits, weights=weights)
    hist = np.array(hist, dtype=float)
    binsize = bin_edges[1] - bin_edges[0]
    extrapoints = len([v for v in a if defaultlimits[0] > v or v > defaultlimits[1]])
    if extrapoints > 0 and printextras:
        warnings.warn('Points outside given histogram range = %s' % extrapoints, stacklevel=3)
    return HistogramResult(hist, defaultlimits[0], binsize, extrapoints)