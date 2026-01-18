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
def yeojohnson_normmax(x, brack=None):
    """Compute optimal Yeo-Johnson transform parameter.

    Compute optimal Yeo-Johnson transform parameter for input data, using
    maximum likelihood estimation.

    Parameters
    ----------
    x : array_like
        Input array.
    brack : 2-tuple, optional
        The starting interval for a downhill bracket search with
        `optimize.brent`. Note that this is in most cases not critical; the
        final result is allowed to be outside this bracket. If None,
        `optimize.fminbound` is used with bounds that avoid overflow.

    Returns
    -------
    maxlog : float
        The optimal transform parameter found.

    See Also
    --------
    yeojohnson, yeojohnson_llf, yeojohnson_normplot

    Notes
    -----
    .. versionadded:: 1.2.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Generate some data and determine optimal ``lmbda``

    >>> rng = np.random.default_rng()
    >>> x = stats.loggamma.rvs(5, size=30, random_state=rng) + 5
    >>> lmax = stats.yeojohnson_normmax(x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> prob = stats.yeojohnson_normplot(x, -10, 10, plot=ax)
    >>> ax.axvline(lmax, color='r')

    >>> plt.show()

    """

    def _neg_llf(lmbda, data):
        llf = yeojohnson_llf(lmbda, data)
        llf[np.isinf(llf)] = -np.inf
        return -llf
    with np.errstate(invalid='ignore'):
        if not np.all(np.isfinite(x)):
            raise ValueError('Yeo-Johnson input must be finite.')
        if np.all(x == 0):
            return 1.0
        if brack is not None:
            return optimize.brent(_neg_llf, brack=brack, args=(x,))
        x = np.asarray(x)
        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
        log1p_max_x = np.log1p(20 * np.max(np.abs(x)))
        log_eps = np.log(np.finfo(dtype).eps)
        log_tiny_float = (np.log(np.finfo(dtype).tiny) - log_eps) / 2
        log_max_float = (np.log(np.finfo(dtype).max) + log_eps) / 2
        lb = log_tiny_float / log1p_max_x
        ub = log_max_float / log1p_max_x
        if np.all(x < 0):
            lb, ub = (2 - ub, 2 - lb)
        elif np.any(x < 0):
            lb, ub = (max(2 - ub, lb), min(2 - lb, ub))
        tol_brent = 1.48e-08
        return optimize.fminbound(_neg_llf, lb, ub, args=(x,), xtol=tol_brent)