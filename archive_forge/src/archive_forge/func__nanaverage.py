import warnings
from functools import partial
from numbers import Integral
import numpy as np
from scipy import linalg, sparse
from ..utils import deprecated
from ..utils._param_validation import Interval, StrOptions, validate_params
from . import check_random_state
from ._array_api import _is_numpy_namespace, device, get_namespace
from .sparsefuncs_fast import csr_row_norms
from .validation import check_array
def _nanaverage(a, weights=None):
    """Compute the weighted average, ignoring NaNs.

    Parameters
    ----------
    a : ndarray
        Array containing data to be averaged.
    weights : array-like, default=None
        An array of weights associated with the values in a. Each value in a
        contributes to the average according to its associated weight. The
        weights array can either be 1-D of the same shape as a. If `weights=None`,
        then all data in a are assumed to have a weight equal to one.

    Returns
    -------
    weighted_average : float
        The weighted average.

    Notes
    -----
    This wrapper to combine :func:`numpy.average` and :func:`numpy.nanmean`, so
    that :func:`np.nan` values are ignored from the average and weights can
    be passed. Note that when possible, we delegate to the prime methods.
    """
    if len(a) == 0:
        return np.nan
    mask = np.isnan(a)
    if mask.all():
        return np.nan
    if weights is None:
        return np.nanmean(a)
    weights = np.asarray(weights)
    a, weights = (a[~mask], weights[~mask])
    try:
        return np.average(a, weights=weights)
    except ZeroDivisionError:
        return np.average(a)