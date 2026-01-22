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
def _normplot(method, x, la, lb, plot=None, N=80):
    """Compute parameters for a Box-Cox or Yeo-Johnson normality plot,
    optionally show it.

    See `boxcox_normplot` or `yeojohnson_normplot` for details.
    """
    if method == 'boxcox':
        title = 'Box-Cox Normality Plot'
        transform_func = boxcox
    else:
        title = 'Yeo-Johnson Normality Plot'
        transform_func = yeojohnson
    x = np.asarray(x)
    if x.size == 0:
        return x
    if lb <= la:
        raise ValueError('`lb` has to be larger than `la`.')
    if method == 'boxcox' and np.any(x <= 0):
        raise ValueError('Data must be positive.')
    lmbdas = np.linspace(la, lb, num=N)
    ppcc = lmbdas * 0.0
    for i, val in enumerate(lmbdas):
        z = transform_func(x, lmbda=val)
        _, (_, _, r) = probplot(z, dist='norm', fit=True)
        ppcc[i] = r
    if plot is not None:
        plot.plot(lmbdas, ppcc, 'x')
        _add_axis_labels_title(plot, xlabel='$\\lambda$', ylabel='Prob Plot Corr. Coef.', title=title)
    return (lmbdas, ppcc)