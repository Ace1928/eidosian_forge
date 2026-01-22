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
def _t_confidence_interval(df, t, confidence_level, alternative):
    if confidence_level < 0 or confidence_level > 1:
        message = '`confidence_level` must be a number between 0 and 1.'
        raise ValueError(message)
    if alternative < 0:
        p = confidence_level
        low, high = np.broadcast_arrays(-np.inf, special.stdtrit(df, p))
    elif alternative > 0:
        p = 1 - confidence_level
        low, high = np.broadcast_arrays(special.stdtrit(df, p), np.inf)
    elif alternative == 0:
        tail_probability = (1 - confidence_level) / 2
        p = (tail_probability, 1 - tail_probability)
        p = np.reshape(p, [2] + [1] * np.asarray(df).ndim)
        low, high = special.stdtrit(df, p)
    else:
        p, nans = np.broadcast_arrays(t, np.nan)
        low, high = (nans, nans)
    return (low[()], high[()])