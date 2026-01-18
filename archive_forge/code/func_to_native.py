import sys
import operator
import numpy as np
from math import prod
import scipy.sparse as sp
from scipy._lib._util import np_long, np_ulong
def to_native(A):
    """
    Ensure that the data type of the NumPy array `A` has native byte order.

    `A` must be a NumPy array.  If the data type of `A` does not have native
    byte order, a copy of `A` with a native byte order is returned. Otherwise
    `A` is returned.
    """
    dt = A.dtype
    if dt.isnative:
        return A
    return np.asarray(A, dtype=dt.newbyteorder('native'))