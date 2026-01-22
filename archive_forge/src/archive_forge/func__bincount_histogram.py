import numpy as np
from ..util.dtype import dtype_range, dtype_limits
from .._shared import utils
def _bincount_histogram(image, source_range, bin_centers=None):
    """
    Efficient histogram calculation for an image of integers.

    This function is significantly more efficient than np.histogram but
    works only on images of integers. It is based on np.bincount.

    Parameters
    ----------
    image : array
        Input image.
    source_range : string
        'image' determines the range from the input image.
        'dtype' determines the range from the expected range of the images
        of that data type.

    Returns
    -------
    hist : array
        The values of the histogram.
    bin_centers : array
        The values at the center of the bins.
    """
    if bin_centers is None:
        bin_centers = _bincount_histogram_centers(image, source_range)
    image_min, image_max = (bin_centers[0], bin_centers[-1])
    image = _offset_array(image, image_min, image_max)
    hist = np.bincount(image.ravel(), minlength=image_max - min(image_min, 0) + 1)
    if source_range == 'image':
        idx = max(image_min, 0)
        hist = hist[idx:]
    return (hist, bin_centers)