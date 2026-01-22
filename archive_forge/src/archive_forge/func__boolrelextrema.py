import math
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx import jit
def _boolrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.

    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated. 'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default 'clip'. See cupy.take.

    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.

    See also
    --------
    argrelmax, argrelmin
    """
    if int(order) != order or order < 1:
        raise ValueError('Order must be an int >= 1')
    if data.ndim < 3:
        results = cupy.empty(data.shape, dtype=bool)
        _peak_finding(data, comparator, axis, order, mode, results)
    else:
        datalen = data.shape[axis]
        locs = cupy.arange(0, datalen)
        results = cupy.ones(data.shape, dtype=bool)
        main = cupy.take(data, locs, axis=axis)
        for shift in cupy.arange(1, order + 1):
            if mode == 'clip':
                p_locs = cupy.clip(locs + shift, a_min=None, a_max=datalen - 1)
                m_locs = cupy.clip(locs - shift, a_min=0, a_max=None)
            else:
                p_locs = locs + shift
                m_locs = locs - shift
            plus = cupy.take(data, p_locs, axis=axis)
            minus = cupy.take(data, m_locs, axis=axis)
            results &= comparator(main, plus)
            results &= comparator(main, minus)
            if ~results.any():
                return results
    return results