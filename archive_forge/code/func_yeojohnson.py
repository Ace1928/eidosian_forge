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
def yeojohnson(x, lmbda=None):
    """Return a dataset transformed by a Yeo-Johnson power transformation.

    Parameters
    ----------
    x : ndarray
        Input array.  Should be 1-dimensional.
    lmbda : float, optional
        If ``lmbda`` is ``None``, find the lambda that maximizes the
        log-likelihood function and return it as the second output argument.
        Otherwise the transformation is done for the given value.

    Returns
    -------
    yeojohnson: ndarray
        Yeo-Johnson power transformed array.
    maxlog : float, optional
        If the `lmbda` parameter is None, the second returned argument is
        the lambda that maximizes the log-likelihood function.

    See Also
    --------
    probplot, yeojohnson_normplot, yeojohnson_normmax, yeojohnson_llf, boxcox

    Notes
    -----
    The Yeo-Johnson transform is given by::

        y = ((x + 1)**lmbda - 1) / lmbda,                for x >= 0, lmbda != 0
            log(x + 1),                                  for x >= 0, lmbda = 0
            -((-x + 1)**(2 - lmbda) - 1) / (2 - lmbda),  for x < 0, lmbda != 2
            -log(-x + 1),                                for x < 0, lmbda = 2

    Unlike `boxcox`, `yeojohnson` does not require the input data to be
    positive.

    .. versionadded:: 1.2.0


    References
    ----------
    I. Yeo and R.A. Johnson, "A New Family of Power Transformations to
    Improve Normality or Symmetry", Biometrika 87.4 (2000):


    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    We generate some random variates from a non-normal distribution and make a
    probability plot for it, to show it is non-normal in the tails:

    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(211)
    >>> x = stats.loggamma.rvs(5, size=500) + 5
    >>> prob = stats.probplot(x, dist=stats.norm, plot=ax1)
    >>> ax1.set_xlabel('')
    >>> ax1.set_title('Probplot against normal distribution')

    We now use `yeojohnson` to transform the data so it's closest to normal:

    >>> ax2 = fig.add_subplot(212)
    >>> xt, lmbda = stats.yeojohnson(x)
    >>> prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
    >>> ax2.set_title('Probplot after Yeo-Johnson transformation')

    >>> plt.show()

    """
    x = np.asarray(x)
    if x.size == 0:
        return x
    if np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError('Yeo-Johnson transformation is not defined for complex numbers.')
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64, copy=False)
    if lmbda is not None:
        return _yeojohnson_transform(x, lmbda)
    lmax = yeojohnson_normmax(x)
    y = _yeojohnson_transform(x, lmax)
    return (y, lmax)