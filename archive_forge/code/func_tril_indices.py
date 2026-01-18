import functools
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy._creation import from_data
from cupy._manipulation import join
def tril_indices(n, k=0, m=None):
    """Returns the indices of the lower triangular matrix.
    Here, the first group of elements contains row coordinates
    of all indices and the second group of elements
    contains column coordinates.

    Parameters
    ----------
    n : int
        The row dimension of the arrays for which the returned
        indices will be valid.
    k : int, optional
        Diagonal above which to zero elements. `k = 0`
        (the default) is the main diagonal, `k < 0` is
        below it and `k > 0` is above.
    m : int, optional
        The column dimension of the arrays for which the
        returned arrays will be valid. By default, `m = n`.

    Returns
    -------
    y : tuple of ndarrays
        The indices for the triangle. The returned tuple
        contains two arrays, each with the indices along
        one dimension of the array.

    See Also
    --------
    numpy.tril_indices

    """
    tri_ = cupy.tri(n, m, k=k, dtype=bool)
    return tuple((cupy.broadcast_to(inds, tri_.shape)[tri_] for inds in cupy.indices(tri_.shape, dtype=int)))