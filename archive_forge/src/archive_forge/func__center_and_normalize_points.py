import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
def _center_and_normalize_points(points):
    """Center and normalize image points.

    The points are transformed in a two-step procedure that is expressed
    as a transformation matrix. The matrix of the resulting points is usually
    better conditioned than the matrix of the original points.

    Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(D).

    If the points are all identical, the returned values will contain nan.

    Parameters
    ----------
    points : (N, D) array
        The coordinates of the image points.

    Returns
    -------
    matrix : (D+1, D+1) array_like
        The transformation matrix to obtain the new points.
    new_points : (N, D) array
        The transformed image points.

    References
    ----------
    .. [1] Hartley, Richard I. "In defense of the eight-point algorithm."
           Pattern Analysis and Machine Intelligence, IEEE Transactions on 19.6
           (1997): 580-593.

    """
    n, d = points.shape
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    rms = np.sqrt(np.sum(centered ** 2) / n)
    if rms == 0:
        return (np.full((d + 1, d + 1), np.nan), np.full_like(points, np.nan))
    norm_factor = np.sqrt(d) / rms
    part_matrix = norm_factor * np.concatenate((np.eye(d), -centroid[:, np.newaxis]), axis=1)
    matrix = np.concatenate((part_matrix, [[0] * d + [1]]), axis=0)
    points_h = np.vstack([points.T, np.ones(n)])
    new_points_h = (matrix @ points_h).T
    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]
    return (matrix, new_points)