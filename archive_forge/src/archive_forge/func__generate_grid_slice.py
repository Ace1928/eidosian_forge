import itertools
import functools
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import _supported_float_type
from ..metrics import mean_squared_error
from ..util import img_as_float
def _generate_grid_slice(shape, *, offset, stride=3):
    """Generate slices of uniformly-spaced points in an array.

    Parameters
    ----------
    shape : tuple of int
        Shape of the mask.
    offset : int
        The offset of the grid of ones. Iterating over ``offset`` will cover
        the entire array. It should be between 0 and ``stride ** ndim``, not
        inclusive, where ``ndim = len(shape)``.
    stride : int, optional
        The spacing between ones, used in each dimension.

    Returns
    -------
    mask : ndarray
        The mask.

    Examples
    --------
    >>> shape = (4, 4)
    >>> array = np.zeros(shape, dtype=int)
    >>> grid_slice = _generate_grid_slice(shape, offset=0, stride=2)
    >>> array[grid_slice] = 1
    >>> print(array)
    [[1 0 1 0]
     [0 0 0 0]
     [1 0 1 0]
     [0 0 0 0]]

    Changing the offset moves the location of the 1s:

    >>> array = np.zeros(shape, dtype=int)
    >>> grid_slice = _generate_grid_slice(shape, offset=3, stride=2)
    >>> array[grid_slice] = 1
    >>> print(array)
    [[0 0 0 0]
     [0 1 0 1]
     [0 0 0 0]
     [0 1 0 1]]
    """
    phases = np.unravel_index(offset, (stride,) * len(shape))
    mask = tuple((slice(p, None, stride) for p in phases))
    return mask