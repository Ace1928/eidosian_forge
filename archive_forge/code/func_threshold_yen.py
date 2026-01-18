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
def threshold_yen(image=None, nbins=256, *, hist=None):
    """Return threshold value based on Yen's method.
    Either image or hist must be provided. In case hist is given, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    hist : array, or 2-tuple of arrays, optional
        Histogram from which to determine the threshold, and optionally a
        corresponding array of bin center intensities.
        An alternative use of this function is to pass it only hist.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion
           for Automatic Multilevel Thresholding" IEEE Trans. on Image
           Processing, 4(3): 370-378. :DOI:`10.1109/83.366472`
    .. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165, :DOI:`10.1117/1.1631315`
           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
    .. [3] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_yen(image)
    >>> binary = image <= thresh
    """
    counts, bin_centers = _validate_image_histogram(image, hist, nbins)
    if bin_centers.size == 1:
        return bin_centers[0]
    pmf = counts.astype('float32', copy=False) / counts.sum()
    P1 = np.cumsum(pmf)
    P1_sq = np.cumsum(pmf ** 2)
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    crit = np.log((P1_sq[:-1] * P2_sq[1:]) ** (-1) * (P1[:-1] * (1.0 - P1[:-1])) ** 2)
    return bin_centers[crit.argmax()]