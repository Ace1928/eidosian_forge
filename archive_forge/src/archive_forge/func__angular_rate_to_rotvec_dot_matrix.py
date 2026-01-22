import numpy as np
from scipy.linalg import solve_banded
from ._rotation import Rotation
def _angular_rate_to_rotvec_dot_matrix(rotvecs):
    """Compute matrices to transform angular rates to rot. vector derivatives.

    The matrices depend on the current attitude represented as a rotation
    vector.

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        Set of rotation vectors.

    Returns
    -------
    ndarray, shape (n, 3, 3)
    """
    norm = np.linalg.norm(rotvecs, axis=1)
    k = np.empty_like(norm)
    mask = norm > 0.0001
    nm = norm[mask]
    k[mask] = (1 - 0.5 * nm / np.tan(0.5 * nm)) / nm ** 2
    mask = ~mask
    nm = norm[mask]
    k[mask] = 1 / 12 + 1 / 720 * nm ** 2
    skew = _create_skew_matrix(rotvecs)
    result = np.empty((len(rotvecs), 3, 3))
    result[:] = np.identity(3)
    result[:] += 0.5 * skew
    result[:] += k[:, None, None] * np.matmul(skew, skew)
    return result