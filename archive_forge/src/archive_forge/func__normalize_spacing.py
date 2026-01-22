from math import sqrt
from numbers import Real
import numpy as np
from scipy import ndimage as ndi
def _normalize_spacing(spacing, ndims):
    """Normalize spacing parameter.

    The `spacing` parameter should be a sequence of numbers matching
    the image dimensions. If `spacing` is a scalar, assume equal
    spacing along all dimensions.

    Parameters
    ---------
    spacing : Any
        User-provided `spacing` keyword.
    ndims : int
        Number of image dimensions.

    Returns
    -------
    spacing : array
        Corrected spacing.

    Raises
    ------
    ValueError
        If `spacing` is invalid.

    """
    spacing = np.array(spacing)
    if spacing.shape == ():
        spacing = np.broadcast_to(spacing, shape=(ndims,))
    elif spacing.shape != (ndims,):
        raise ValueError(f"spacing isn't a scalar nor a sequence of shape {(ndims,)}, got {spacing}.")
    if not all((isinstance(s, Real) for s in spacing)):
        raise TypeError(f"Element of spacing isn't float or integer type, got {spacing}.")
    if not all(np.isfinite(spacing)):
        raise ValueError(f'Invalid spacing parameter. All elements must be finite, got {spacing}.')
    return spacing