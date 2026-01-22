import math
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx import jit
def _unpack_condition_args(interval, x, peaks):
    """
    Parse condition arguments for `find_peaks`.

    Parameters
    ----------
    interval : number or ndarray or sequence
        Either a number or ndarray or a 2-element sequence of the former. The
        first value is always interpreted as `imin` and the second,
        if supplied, as `imax`.
    x : ndarray
        The signal with `peaks`.
    peaks : ndarray
        An array with indices used to reduce `imin` and / or `imax` if those
        are arrays.

    Returns
    -------
    imin, imax : number or ndarray or None
        Minimal and maximal value in `argument`.

    Raises
    ------
    ValueError :
        If interval border is given as array and its size does not match the
        size of `x`.
    """
    try:
        imin, imax = interval
    except (TypeError, ValueError):
        imin, imax = (interval, None)
    if isinstance(imin, cupy.ndarray):
        if imin.size != x.size:
            raise ValueError('array size of lower interval border must match x')
        imin = imin[peaks]
    if isinstance(imax, cupy.ndarray):
        if imax.size != x.size:
            raise ValueError('array size of upper interval border must match x')
        imax = imax[peaks]
    return (imin, imax)