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
def _alexandergovern_input_validation(samples, nan_policy):
    if len(samples) < 2:
        raise TypeError(f'2 or more inputs required, got {len(samples)}')
    samples = [np.asarray(sample, dtype=float) for sample in samples]
    for i, sample in enumerate(samples):
        if np.size(sample) <= 1:
            raise ValueError('Input sample size must be greater than one.')
        if sample.ndim != 1:
            raise ValueError('Input samples must be one-dimensional')
        if np.isinf(sample).any():
            raise ValueError('Input samples must be finite.')
        contains_nan, nan_policy = _contains_nan(sample, nan_policy=nan_policy)
        if contains_nan and nan_policy == 'omit':
            samples[i] = ma.masked_invalid(sample)
    return samples