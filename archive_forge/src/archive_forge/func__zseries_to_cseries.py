import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def _zseries_to_cseries(zs):
    """Convert z-series to a Chebyshev series.

    Convert a z series to the equivalent Chebyshev series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    zs : 1-D ndarray
        Odd length symmetric z-series, ordered from  low to high.

    Returns
    -------
    c : 1-D ndarray
        Chebyshev coefficients, ordered from  low to high.

    """
    n = (zs.size + 1) // 2
    c = zs[n - 1:].copy()
    c[1:n] *= 2
    return c