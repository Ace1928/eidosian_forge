import numpy as np
from scipy.linalg import solve_banded
from ._rotation import Rotation
def _compute_angular_rate(rotvecs, rotvecs_dot):
    """Compute angular rates given rotation vectors and its derivatives.

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        Set of rotation vectors.
    rotvecs_dot : ndarray, shape (n, 3)
        Set of rotation vector derivatives.

    Returns
    -------
    ndarray, shape (n, 3)
    """
    return _matrix_vector_product_of_stacks(_rotvec_dot_to_angular_rate_matrix(rotvecs), rotvecs_dot)