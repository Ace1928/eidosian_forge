from warnings import warn
import numpy as np
import scipy.ndimage as ndi
from .. import measure
from .._shared.coord import ensure_spacing
def _get_excluded_border_width(image, min_distance, exclude_border):
    """Return border_width values relative to a min_distance if requested."""
    if isinstance(exclude_border, bool):
        border_width = (min_distance if exclude_border else 0,) * image.ndim
    elif isinstance(exclude_border, int):
        if exclude_border < 0:
            raise ValueError('`exclude_border` cannot be a negative value')
        border_width = (exclude_border,) * image.ndim
    elif isinstance(exclude_border, tuple):
        if len(exclude_border) != image.ndim:
            raise ValueError('`exclude_border` should have the same length as the dimensionality of the image.')
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError('`exclude_border`, when expressed as a tuple, must only contain ints.')
            if exclude < 0:
                raise ValueError('`exclude_border` can not be a negative value')
        border_width = exclude_border
    else:
        raise TypeError('`exclude_border` must be bool, int, or tuple with the same length as the dimensionality of the image.')
    return border_width