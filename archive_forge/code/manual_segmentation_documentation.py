from functools import reduce
import numpy as np
from ..draw import polygon
from .._shared.version_requirements import require
Return a label image based on freeform selections made with the mouse.

    Parameters
    ----------
    image : (M, N[, 3]) array
        Grayscale or RGB image.

    alpha : float, optional
        Transparency value for polygons drawn over the image.

    return_all : bool, optional
        If True, an array containing each separate polygon drawn is returned.
        (The polygons may overlap.) If False (default), latter polygons
        "overwrite" earlier ones where they overlap.

    Returns
    -------
    labels : array of int, shape ([Q, ]M, N)
        The segmented regions. If mode is `'separate'`, the leading dimension
        of the array corresponds to the number of regions that the user drew.

    Notes
    -----
    Press and hold the left mouse button to draw around each object.

    Examples
    --------
    >>> from skimage import data, future, io
    >>> camera = data.camera()
    >>> mask = future.manual_lasso_segmentation(camera)  # doctest: +SKIP
    >>> io.imshow(mask)  # doctest: +SKIP
    >>> io.show()  # doctest: +SKIP
    