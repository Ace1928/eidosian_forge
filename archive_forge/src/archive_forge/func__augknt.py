import operator
from numpy.core.multiarray import normalize_axis_index
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.interpolate._bspline import (
def _augknt(x, k):
    """Construct a knot vector appropriate for the order-k interpolation."""
    return cupy.r_[(x[0],) * k, x, (x[-1],) * k]