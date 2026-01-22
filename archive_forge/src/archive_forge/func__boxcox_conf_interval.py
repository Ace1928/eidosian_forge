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
def _boxcox_conf_interval(x, lmax, alpha):
    fac = 0.5 * distributions.chi2.ppf(1 - alpha, 1)
    target = boxcox_llf(lmax, x) - fac

    def rootfunc(lmbda, data, target):
        return boxcox_llf(lmbda, data) - target
    newlm = lmax + 0.5
    N = 0
    while rootfunc(newlm, x, target) > 0.0 and N < 500:
        newlm += 0.1
        N += 1
    if N == 500:
        raise RuntimeError('Could not find endpoint.')
    lmplus = optimize.brentq(rootfunc, lmax, newlm, args=(x, target))
    newlm = lmax - 0.5
    N = 0
    while rootfunc(newlm, x, target) > 0.0 and N < 500:
        newlm -= 0.1
        N += 1
    if N == 500:
        raise RuntimeError('Could not find endpoint.')
    lmminus = optimize.brentq(rootfunc, newlm, lmax, args=(x, target))
    return (lmminus, lmplus)