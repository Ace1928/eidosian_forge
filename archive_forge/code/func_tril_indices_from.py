import functools
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy._creation import from_data
from cupy._manipulation import join
def tril_indices_from(arr, k=0):
    """Returns the indices for the lower-triangle of arr.

    Parameters
    ----------
    arr : cupy.ndarray
          The indices are valid for square arrays
          whose dimensions are the same as arr.
    k : int, optional
        Diagonal offset.

    See Also
    --------
    numpy.tril_indices_from

    """
    if arr.ndim != 2:
        raise ValueError('input array must be 2-d')
    return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1])