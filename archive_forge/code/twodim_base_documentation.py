import functools
import operator
from numpy.core.numeric import (
from numpy.core.overrides import set_array_function_like_doc, set_module
from numpy.core import overrides
from numpy.core import iinfo
from numpy.lib.stride_tricks import broadcast_to

    Return the indices for the upper-triangle of arr.

    See `triu_indices` for full details.

    Parameters
    ----------
    arr : ndarray, shape(N, N)
        The indices will be valid for square arrays.
    k : int, optional
        Diagonal offset (see `triu` for details).

    Returns
    -------
    triu_indices_from : tuple, shape(2) of ndarray, shape(N)
        Indices for the upper-triangle of `arr`.

    Examples
    --------

    Create a 4 by 4 array.

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Pass the array to get the indices of the upper triangular elements.

    >>> triui = np.triu_indices_from(a)
    >>> triui
    (array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]), array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    >>> a[triui]
    array([ 0,  1,  2,  3,  5,  6,  7, 10, 11, 15])

    This is syntactic sugar for triu_indices().

    >>> np.triu_indices(a.shape[0])
    (array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]), array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    Use the `k` parameter to return the indices for the upper triangular array
    from the k-th diagonal.

    >>> triuim1 = np.triu_indices_from(a, k=1)
    >>> a[triuim1]
    array([ 1,  2,  3,  6,  7, 11])


    See Also
    --------
    triu_indices, triu, tril_indices_from

    Notes
    -----
    .. versionadded:: 1.4.0

    