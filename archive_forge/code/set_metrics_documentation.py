import warnings
import numpy as np
from scipy.spatial import cKDTree
Returns pair of points that are Hausdorff distance apart between nonzero
    elements of given images.

    The Hausdorff distance [1]_ is the maximum distance between any point on
    ``image0`` and its nearest point on ``image1``, and vice-versa.

    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a point that is included in a
        set of points. Both arrays must have the same shape.

    Returns
    -------
    point_a, point_b : array
        A pair of points that have Hausdorff distance between them.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance

    Examples
    --------
    >>> points_a = (3, 0)
    >>> points_b = (6, 0)
    >>> shape = (7, 1)
    >>> image_a = np.zeros(shape, dtype=bool)
    >>> image_b = np.zeros(shape, dtype=bool)
    >>> image_a[points_a] = True
    >>> image_b[points_b] = True
    >>> hausdorff_pair(image_a, image_b)
    (array([3, 0]), array([6, 0]))

    