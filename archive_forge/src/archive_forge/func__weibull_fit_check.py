from __future__ import annotations
import math
import warnings
from collections import namedtuple
import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
from ._ansari_swilk_statistics import gscale, swilk
from . import _stats_py
from ._fit import FitResult
from ._stats_py import find_repeats, _normtest_finish, SignificanceResult
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._hypotests import _get_wilcoxon_distr
from ._axis_nan_policy import _axis_nan_policy_factory
def _weibull_fit_check(params, x):
    n = len(x)
    m, u, s = params

    def dnllf_dm(m, u):
        xu = x - u
        return 1 / m - (xu ** m * np.log(xu)).sum() / (xu ** m).sum() + np.log(xu).sum() / n

    def dnllf_du(m, u):
        xu = x - u
        return (m - 1) / m * (xu ** (-1)).sum() - n * (xu ** (m - 1)).sum() / (xu ** m).sum()

    def get_scale(m, u):
        return ((x - u) ** m / n).sum() ** (1 / m)

    def dnllf(params):
        return [dnllf_dm(*params), dnllf_du(*params)]
    suggestion = 'Maximum likelihood estimation is known to be challenging for the three-parameter Weibull distribution. Consider performing a custom goodness-of-fit test using `scipy.stats.monte_carlo_test`.'
    if np.allclose(u, np.min(x)) or m < 1:
        message = 'Maximum likelihood estimation has converged to a solution in which the location is equal to the minimum of the data, the shape parameter is less than 2, or both. The table of critical values in [7] does not include this case. ' + suggestion
        raise ValueError(message)
    try:
        with np.errstate(over='raise', invalid='raise'):
            res = optimize.root(dnllf, params[:-1])
        message = f'Solution of MLE first-order conditions failed: {res.message}. `anderson` cannot continue. ' + suggestion
        if not res.success:
            raise ValueError(message)
    except (FloatingPointError, ValueError) as e:
        message = 'An error occurred while fitting the Weibull distribution to the data, so `anderson` cannot continue. ' + suggestion
        raise ValueError(message) from e
    m, u = res.x
    s = get_scale(m, u)
    return (m, u, s)