import numpy as np
from scipy import ndimage as ndi
from ..._shared.utils import check_nD, warn
from ...morphology.footprints import _footprint_is_sequence
from ...util import img_as_ubyte
from . import generic_cy
def windowed_histogram(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, n_bins=None):
    """Normalized sliding window histogram

    Parameters
    ----------
    image : 2-D array (integer or float)
        Input image.
    footprint : 2-D array (integer or float)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer or float), optional
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    n_bins : int or None
        The number of histogram bins. Will default to ``image.max() + 1``
        if None is passed.

    Returns
    -------
    out : 3-D array (float)
        Array of dimensions (H,W,N), where (H,W) are the dimensions of the
        input image and N is n_bins or ``image.max() + 1`` if no value is
        provided as a parameter. Effectively, each pixel is a N-D feature
        vector that is the histogram. The sum of the elements in the feature
        vector will be 1, unless no pixels in the window were covered by both
        footprint and mask, in which case all elements will be 0.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters.rank import windowed_histogram
    >>> from skimage.morphology import disk, ball
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> hist_img = windowed_histogram(img, disk(5))

    """
    if n_bins is None:
        n_bins = int(image.max()) + 1
    return _apply_vector_per_pixel(generic_cy._windowed_hist, image, footprint, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y, out_dtype=np.float64, pixel_size=n_bins)