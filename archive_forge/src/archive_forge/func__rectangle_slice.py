import numpy as np
from .._shared._geometry import polygon_clip
from .._shared.version_requirements import require
from .._shared.compat import NP_COPY_IF_NEEDED
from ._draw import (
def _rectangle_slice(start, end=None, extent=None):
    """Return the slice ``(top_left, bottom_right)`` of the rectangle.

    Returns
    -------
    (top_left, bottom_right)
        The slice you would need to select the region in the rectangle defined
        by the parameters.
        Select it like:

        ``rect[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]``
    """
    if end is None and extent is None:
        raise ValueError('Either `end` or `extent` must be given.')
    if end is not None and extent is not None:
        raise ValueError('Cannot provide both `end` and `extent`.')
    if extent is not None:
        end = np.asarray(start) + np.asarray(extent)
    top_left = np.minimum(start, end)
    bottom_right = np.maximum(start, end)
    top_left = np.round(top_left).astype(int)
    bottom_right = np.round(bottom_right).astype(int)
    if extent is None:
        bottom_right += 1
    return (top_left, bottom_right)