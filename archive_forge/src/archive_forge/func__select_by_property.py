import math
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx import jit
def _select_by_property(peak_properties, pmin, pmax):
    """
    Evaluate where the generic property of peaks confirms to an interval.

    Parameters
    ----------
    peak_properties : ndarray
        An array with properties for each peak.
    pmin : None or number or ndarray
        Lower interval boundary for `peak_properties`. ``None``
        is interpreted as an open border.
    pmax : None or number or ndarray
        Upper interval boundary for `peak_properties`. ``None``
        is interpreted as an open border.

    Returns
    -------
    keep : bool
        A boolean mask evaluating to true where `peak_properties` confirms
        to the interval.

    See Also
    --------
    find_peaks

    """
    keep = cupy.ones(peak_properties.size, dtype=bool)
    if pmin is not None:
        keep &= pmin <= peak_properties
    if pmax is not None:
        keep &= peak_properties <= pmax
    return keep