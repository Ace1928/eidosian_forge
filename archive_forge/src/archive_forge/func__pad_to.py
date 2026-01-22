import numpy as np
from scipy.stats import entropy
from ..util.dtype import dtype_range
from .._shared.utils import _supported_float_type, check_shape_equality, warn
def _pad_to(arr, shape):
    """Pad an array with trailing zeros to a given target shape.

    Parameters
    ----------
    arr : ndarray
        The input array.
    shape : tuple
        The target shape.

    Returns
    -------
    padded : ndarray
        The padded array.

    Examples
    --------
    >>> _pad_to(np.ones((1, 1), dtype=int), (1, 3))
    array([[1, 0, 0]])
    """
    if not all((s >= i for s, i in zip(shape, arr.shape))):
        raise ValueError(f'Target shape {shape} cannot be smaller than inputshape {arr.shape} along any axis.')
    padding = [(0, s - i) for s, i in zip(shape, arr.shape)]
    return np.pad(arr, pad_width=padding, mode='constant', constant_values=0)