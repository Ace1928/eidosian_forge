import numpy as np
from scipy import ndimage as ndi
from ..._shared.utils import check_nD, warn
from ...morphology.footprints import _footprint_is_sequence
from ...util import img_as_ubyte
from . import generic_cy
def _apply_vector_per_pixel(func, image, footprint, out, mask, shift_x, shift_y, out_dtype=None, pixel_size=1):
    """

    Parameters
    ----------
    func : function
        Cython function to apply.
    image : 2-D array (integer or float)
        Input image.
    footprint : 2-D array (integer or float)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer or float)
        If None, a new array is allocated.
    mask : ndarray (integer or float)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    out_dtype : data-type, optional
        Desired output data-type. Default is None, which means we cast output
        in input dtype.
    pixel_size : int, optional
        Dimension of each pixel.

    Returns
    -------
    out : 3-D array with float dtype of dimensions (H,W,N), where (H,W) are
        the dimensions of the input image and N is n_bins or
        ``image.max() + 1`` if no value is provided as a parameter.
        Effectively, each pixel is a N-D feature vector that is the histogram.
        The sum of the elements in the feature vector will be 1, unless no
        pixels in the window were covered by both footprint and mask, in which
        case all elements will be 0.

    """
    image, footprint, out, mask, n_bins = _preprocess_input(image, footprint, out, mask, out_dtype, pixel_size, shift_x=shift_x, shift_y=shift_y)
    func(image, footprint, shift_x=shift_x, shift_y=shift_y, mask=mask, out=out, n_bins=n_bins)
    return out