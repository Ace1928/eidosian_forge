import math
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx import jit
def argrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.

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
        to consider ``comparator(n, n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated.
        Available options are 'wrap' (wrap around) or 'clip' (treat overflow
        as the same as the last (or first) element).
        Default 'clip'. See cupy.take.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
        return value is a tuple even when `data` is one-dimensional.

    See Also
    --------
    argrelmin, argrelmax

    Examples
    --------
    >>> from cupyx.scipy.signal import argrelextrema
    >>> import cupy
    >>> x = cupy.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelextrema(x, cupy.greater)
    (array([3, 6]),)
    >>> y = cupy.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelextrema(y, cupy.less, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    data = cupy.asarray(data)
    results = _boolrelextrema(data, comparator, axis, order, mode)
    if mode == 'raise':
        raise NotImplementedError("CuPy `take` doesn't support `mode='raise'`.")
    return cupy.nonzero(results)