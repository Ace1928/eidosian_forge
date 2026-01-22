import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
def _setup_constraint_matrix(self, src, dst):
    """Setup and solve the homogeneous epipolar constraint matrix::

            dst' * F * src = 0.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        F_normalized : (3, 3) array
            The normalized solution to the homogeneous system. If the system
            is not well-conditioned, this matrix contains NaNs.
        src_matrix : (3, 3) array
            The transformation matrix to obtain the normalized source
            coordinates.
        dst_matrix : (3, 3) array
            The transformation matrix to obtain the normalized destination
            coordinates.

        """
    src = np.asarray(src)
    dst = np.asarray(dst)
    if src.shape != dst.shape:
        raise ValueError('src and dst shapes must be identical.')
    if src.shape[0] < 8:
        raise ValueError('src.shape[0] must be equal or larger than 8.')
    try:
        src_matrix, src = _center_and_normalize_points(src)
        dst_matrix, dst = _center_and_normalize_points(dst)
    except ZeroDivisionError:
        self.params = np.full((3, 3), np.nan)
        return 3 * [np.full((3, 3), np.nan)]
    A = np.ones((src.shape[0], 9))
    A[:, :2] = src
    A[:, :3] *= dst[:, 0, np.newaxis]
    A[:, 3:5] = src
    A[:, 3:6] *= dst[:, 1, np.newaxis]
    A[:, 6:8] = src
    _, _, V = np.linalg.svd(A)
    F_normalized = V[-1, :].reshape(3, 3)
    return (F_normalized, src_matrix, dst_matrix)