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
def threshold_isodata(image=None, nbins=256, return_all=False, *, hist=None):
    """Return threshold value(s) based on ISODATA method.

    Histogram-based threshold, known as Ridler-Calvard method or inter-means.
    Threshold values returned satisfy the following equality::

        threshold = (image[image <= threshold].mean() +
                     image[image > threshold].mean()) / 2.0

    That is, returned thresholds are intensities that separate the image into
    two groups of pixels, where the threshold intensity is midway between the
    mean intensities of these groups.

    For integer images, the above equality holds to within one; for floating-
    point images, the equality holds to within the histogram bin-width.

    Either image or hist must be provided. In case hist is given, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    return_all : bool, optional
        If False (default), return only the lowest threshold that satisfies
        the above equality. If True, return all valid thresholds.
    hist : array, or 2-tuple of arrays, optional
        Histogram to determine the threshold from and a corresponding array
        of bin center intensities. Alternatively, only the histogram can be
        passed.

    Returns
    -------
    threshold : float or int or array
        Threshold value(s).

    References
    ----------
    .. [1] Ridler, TW & Calvard, S (1978), "Picture thresholding using an
           iterative selection method"
           IEEE Transactions on Systems, Man and Cybernetics 8: 630-632,
           :DOI:`10.1109/TSMC.1978.4310039`
    .. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165,
           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
           :DOI:`10.1117/1.1631315`
    .. [3] ImageJ AutoThresholder code,
           http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import coins
    >>> image = coins()
    >>> thresh = threshold_isodata(image)
    >>> binary = image > thresh
    """
    counts, bin_centers = _validate_image_histogram(image, hist, nbins)
    if len(bin_centers) == 1:
        if return_all:
            return bin_centers
        else:
            return bin_centers[0]
    counts = counts.astype('float32', copy=False)
    csuml = np.cumsum(counts)
    csumh = csuml[-1] - csuml
    intensity_sum = counts * bin_centers
    csum_intensity = np.cumsum(intensity_sum)
    lower = csum_intensity[:-1] / csuml[:-1]
    higher = (csum_intensity[-1] - csum_intensity[:-1]) / csumh[:-1]
    all_mean = (lower + higher) / 2.0
    bin_width = bin_centers[1] - bin_centers[0]
    distances = all_mean - bin_centers[:-1]
    thresholds = bin_centers[:-1][(distances >= 0) & (distances < bin_width)]
    if return_all:
        return thresholds
    else:
        return thresholds[0]