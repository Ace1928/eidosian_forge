import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import _supported_float_type
from ..morphology import dilation, erosion, square
from ..util import img_as_float, view_as_windows
from ..color import gray2rgb
Return image with boundaries between labeled regions highlighted.

    Parameters
    ----------
    image : (M, N[, 3]) array
        Grayscale or RGB image.
    label_img : (M, N) array of int
        Label array where regions are marked by different integer values.
    color : length-3 sequence, optional
        RGB color of boundaries in the output image.
    outline_color : length-3 sequence, optional
        RGB color surrounding boundaries in the output image. If None, no
        outline is drawn.
    mode : string in {'thick', 'inner', 'outer', 'subpixel'}, optional
        The mode for finding boundaries.
    background_label : int, optional
        Which label to consider background (this is only useful for
        modes ``inner`` and ``outer``).

    Returns
    -------
    marked : (M, N, 3) array of float
        An image in which the boundaries between labels are
        superimposed on the original image.

    See Also
    --------
    find_boundaries
    