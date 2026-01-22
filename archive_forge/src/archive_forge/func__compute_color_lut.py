import functools
from math import ceil
import numbers
import scipy.stats
import numpy as np
from ..util.dtype import img_as_float
from .._shared import utils
from .._shared.utils import _supported_float_type, warn
from ._denoise_cy import _denoise_bilateral, _denoise_tv_bregman
from .. import color
from ..color.colorconv import ycbcr_from_rgb
def _compute_color_lut(bins, sigma, max_value, *, dtype=float):
    """Helping function. Define a lookup table containing Gaussian filter
    values using the color distance sigma.

    Parameters
    ----------
    bins : int
        Number of discrete values for Gaussian weights of color filtering.
        A larger value results in improved accuracy.
    sigma : float
        Standard deviation for grayvalue/color distance (radiometric
        similarity). A larger value results in averaging of pixels with larger
        radiometric differences. Note, that the image will be converted using
        the `img_as_float` function and thus the standard deviation is in
        respect to the range ``[0, 1]``. If the value is ``None`` the standard
        deviation of the ``image`` will be used.
    max_value : float
        Maximum value of the input image.
    dtype : data type object, optional (default : float)
        The type and size of the data to be returned.

    Returns
    -------
    color_lut : ndarray
        Lookup table for the color distance sigma.
    """
    values = np.linspace(0, max_value, bins, endpoint=False)
    return _gaussian_weight(values, sigma ** 2, dtype=dtype)