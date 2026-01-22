from itertools import product
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from ..measure.pnpoly import grid_points_in_poly
from ._convex_hull import possible_hull
from ..measure._label import label
from ..util import unique_rows
from .._shared.utils import warn
Compute the convex hull image of individual objects in a binary image.

    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.

    Parameters
    ----------
    image : (M, N) ndarray
        Binary input image.
    connectivity : {1, 2}, int, optional
        Determines the neighbors of each pixel. Adjacent elements
        within a squared distance of ``connectivity`` from pixel center
        are considered neighbors.::

            1-connectivity      2-connectivity
                  [ ]           [ ]  [ ]  [ ]
                   |               \  |  /
             [ ]--[x]--[ ]      [ ]--[x]--[ ]
                   |               /  |  \
                  [ ]           [ ]  [ ]  [ ]

    Returns
    -------
    hull : ndarray of bool
        Binary image with pixels inside convex hull set to ``True``.

    Notes
    -----
    This function uses ``skimage.morphology.label`` to define unique objects,
    finds the convex hull of each using ``convex_hull_image``, and combines
    these regions with logical OR. Be aware the convex hulls of unconnected
    objects may overlap in the result. If this is suspected, consider using
    convex_hull_image separately on each object or adjust ``connectivity``.
    