import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def _zseries_div(z1, z2):
    """Divide the first z-series by the second.

    Divide `z1` by `z2` and return the quotient and remainder as z-series.
    Warning: this implementation only applies when both z1 and z2 have the
    same symmetry, which is sufficient for present purposes.

    Parameters
    ----------
    z1, z2 : 1-D ndarray
        The arrays must be 1-D and have the same symmetry, but this is not
        checked.

    Returns
    -------

    (quotient, remainder) : 1-D ndarrays
        Quotient and remainder as z-series.

    Notes
    -----
    This is not the same as polynomial division on account of the desired form
    of the remainder. If symmetric/anti-symmetric z-series are denoted by S/A
    then the following rules apply:

    S/S -> S,S
    A/A -> S,A

    The restriction to types of the same symmetry could be fixed but seems like
    unneeded generality. There is no natural form for the remainder in the case
    where there is no symmetry.

    """
    z1 = z1.copy()
    z2 = z2.copy()
    lc1 = len(z1)
    lc2 = len(z2)
    if lc2 == 1:
        z1 /= z2
        return (z1, z1[:1] * 0)
    elif lc1 < lc2:
        return (z1[:1] * 0, z1)
    else:
        dlen = lc1 - lc2
        scl = z2[0]
        z2 /= scl
        quo = np.empty(dlen + 1, dtype=z1.dtype)
        i = 0
        j = dlen
        while i < j:
            r = z1[i]
            quo[i] = z1[i]
            quo[dlen - i] = r
            tmp = r * z2
            z1[i:i + lc2] -= tmp
            z1[j:j + lc2] -= tmp
            i += 1
            j -= 1
        r = z1[i]
        quo[i] = r
        tmp = r * z2
        z1[i:i + lc2] -= tmp
        quo /= scl
        rem = z1[i + 1:i - 1 + lc2].copy()
        return (quo, rem)