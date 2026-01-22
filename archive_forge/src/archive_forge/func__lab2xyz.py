from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
def _lab2xyz(lab, illuminant, observer):
    """Convert CIE-LAB to XYZ color space.

    Internal function for :func:`~.lab2xyz` and others. In addition to the
    converted image, return the number of invalid pixels in the Z channel for
    correct warning propagation.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in XYZ format. Same dimensions as input.
    n_invalid : int
        Number of invalid pixels in the Z channel after conversion.
    """
    arr = _prepare_colorarray(lab, channel_axis=-1).copy()
    L, a, b = (arr[..., 0], arr[..., 1], arr[..., 2])
    y = (L + 16.0) / 116.0
    x = a / 500.0 + y
    z = y - b / 200.0
    invalid = np.atleast_1d(z < 0).nonzero()
    n_invalid = invalid[0].size
    if n_invalid != 0:
        if z.ndim > 0:
            z[invalid] = 0
        else:
            z = 0
    out = np.stack([x, y, z], axis=-1)
    mask = out > 0.2068966
    out[mask] = np.power(out[mask], 3.0)
    out[~mask] = (out[~mask] - 16.0 / 116.0) / 7.787
    xyz_ref_white = xyz_tristimulus_values(illuminant=illuminant, observer=observer)
    out *= xyz_ref_white
    return (out, n_invalid)