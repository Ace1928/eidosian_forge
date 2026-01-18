import inspect
import itertools
import math
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
from scipy import ndimage as ndi
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, warn
from .._shared.version_requirements import require
from ..exposure import histogram
from ..filters._multiotsu import (
from ..transform import integral_image
from ..util import dtype_limits
from ._sparse import _correlate_sparse, _validate_window_size
def threshold_triangle(image, nbins=256):
    """Return threshold value based on the triangle algorithm.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Zack, G. W., Rogers, W. E. and Latt, S. A., 1977,
       Automatic Measurement of Sister Chromatid Exchange Frequency,
       Journal of Histochemistry and Cytochemistry 25 (7), pp. 741-753
       :DOI:`10.1177/25.7.70454`
    .. [2] ImageJ AutoThresholder code,
       http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_triangle(image)
    >>> binary = image > thresh
    """
    hist, bin_centers = histogram(image.reshape(-1), nbins, source_range='image')
    nbins = len(hist)
    arg_peak_height = np.argmax(hist)
    peak_height = hist[arg_peak_height]
    arg_low_level, arg_high_level = np.flatnonzero(hist)[[0, -1]]
    if arg_low_level == arg_high_level:
        return image.ravel()[0]
    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = hist[::-1]
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1
    del arg_high_level
    width = arg_peak_height - arg_low_level
    x1 = np.arange(width)
    y1 = hist[x1 + arg_low_level]
    norm = np.sqrt(peak_height ** 2 + width ** 2)
    peak_height /= norm
    width /= norm
    length = peak_height * x1 - width * y1
    arg_level = np.argmax(length) + arg_low_level
    if flip:
        arg_level = nbins - arg_level - 1
    return bin_centers[arg_level]