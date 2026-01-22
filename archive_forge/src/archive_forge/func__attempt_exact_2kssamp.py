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
def _attempt_exact_2kssamp(n1, n2, g, d, alternative):
    """Attempts to compute the exact 2sample probability.

    n1, n2 are the sample sizes
    g is the gcd(n1, n2)
    d is the computed max difference in ECDFs

    Returns (success, d, probability)
    """
    lcm = n1 // g * n2
    h = int(np.round(d * lcm))
    d = h * 1.0 / lcm
    if h == 0:
        return (True, d, 1.0)
    saw_fp_error, prob = (False, np.nan)
    try:
        with np.errstate(invalid='raise', over='raise'):
            if alternative == 'two-sided':
                if n1 == n2:
                    prob = _compute_prob_outside_square(n1, h)
                else:
                    prob = _compute_outer_prob_inside_method(n1, n2, g, h)
            elif n1 == n2:
                jrange = np.arange(h)
                prob = np.prod((n1 - jrange) / (n1 + jrange + 1.0))
            else:
                with np.errstate(over='raise'):
                    num_paths = _count_paths_outside_method(n1, n2, g, h)
                bin = special.binom(n1 + n2, n1)
                if num_paths > bin or np.isinf(bin):
                    saw_fp_error = True
                else:
                    prob = num_paths / bin
    except (FloatingPointError, OverflowError):
        saw_fp_error = True
    if saw_fp_error:
        return (False, d, np.nan)
    if not 0 <= prob <= 1:
        return (False, d, prob)
    return (True, d, prob)