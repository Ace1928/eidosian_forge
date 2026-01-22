import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import _supported_float_type
from ..morphology import dilation, erosion, square
from ..util import img_as_float, view_as_windows
from ..color import gray2rgb
def _find_boundaries_subpixel(label_img):
    """See ``find_boundaries(..., mode='subpixel')``.

    Notes
    -----
    This function puts in an empty row and column between each *actual*
    row and column of the image, for a corresponding shape of ``2s - 1``
    for every image dimension of size ``s``. These "interstitial" rows
    and columns are filled as ``True`` if they separate two labels in
    `label_img`, ``False`` otherwise.

    I used ``view_as_windows`` to get the neighborhood of each pixel.
    Then I check whether there are two labels or more in that
    neighborhood.
    """
    ndim = label_img.ndim
    max_label = np.iinfo(label_img.dtype).max
    label_img_expanded = np.zeros([2 * s - 1 for s in label_img.shape], label_img.dtype)
    pixels = (slice(None, None, 2),) * ndim
    label_img_expanded[pixels] = label_img
    edges = np.ones(label_img_expanded.shape, dtype=bool)
    edges[pixels] = False
    label_img_expanded[edges] = max_label
    windows = view_as_windows(np.pad(label_img_expanded, 1, mode='edge'), (3,) * ndim)
    boundaries = np.zeros_like(edges)
    for index in np.ndindex(label_img_expanded.shape):
        if edges[index]:
            values = np.unique(windows[index].ravel())
            if len(values) > 2:
                boundaries[index] = True
    return boundaries