import numpy as np
from scipy.stats import entropy
from ..util.dtype import dtype_range
from .._shared.utils import _supported_float_type, check_shape_equality, warn
def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = _supported_float_type((image0.dtype, image1.dtype))
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return (image0, image1)