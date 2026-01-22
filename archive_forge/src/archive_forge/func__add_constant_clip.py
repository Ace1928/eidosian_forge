import numpy as np
from .._shared.utils import warn
from ..util import dtype_limits, invert, crop
from . import grayreconstruct, _util
from ._extrema_cy import _local_maxima
def _add_constant_clip(image, const_value):
    """Add constant to the image while handling overflow issues gracefully."""
    min_dtype, max_dtype = dtype_limits(image, clip_negative=False)
    if const_value > max_dtype - min_dtype:
        raise ValueError('The added constant is not compatiblewith the image data type.')
    result = image + const_value
    result[image > max_dtype - const_value] = max_dtype
    return result