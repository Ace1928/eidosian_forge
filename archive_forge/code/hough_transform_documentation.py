import numpy as np
from scipy.spatial import cKDTree
from ._hough_transform import _hough_circle, _hough_ellipse, _hough_line
from ._hough_transform import _probabilistic_hough_line as _prob_hough_line
Keep points that are separated by certain distance in each dimension.

    The first point is always accepted and all subsequent points are selected
    so that they are distant from all their preceding ones.

    Parameters
    ----------
    xs : array, shape (M,)
        X coordinates of points.
    ys : array, shape (M,)
        Y coordinates of points.
    min_xdistance : int
        Minimum distance separating points in the x dimension.
    min_ydistance : int
        Minimum distance separating points in the y dimension.
    max_points : int
        Max number of distant points to keep.

    Returns
    -------
    should_keep : array of bool
        A mask array for distant points to keep.
    