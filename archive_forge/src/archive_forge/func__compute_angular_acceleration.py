import numpy as np
from scipy.linalg import solve_banded
from ._rotation import Rotation
def _compute_angular_acceleration(rotvecs, rotvecs_dot, rotvecs_dot_dot):
    """Compute angular acceleration given rotation vector and its derivatives.

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        Set of rotation vectors.
    rotvecs_dot : ndarray, shape (n, 3)
        Set of rotation vector derivatives.
    rotvecs_dot_dot : ndarray, shape (n, 3)
        Set of rotation vector second derivatives.

    Returns
    -------
    ndarray, shape (n, 3)
    """
    return _compute_angular_rate(rotvecs, rotvecs_dot_dot) + _angular_acceleration_nonlinear_term(rotvecs, rotvecs_dot)