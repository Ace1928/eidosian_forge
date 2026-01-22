import numpy as np
from scipy.linalg import solve_banded
from ._rotation import Rotation
def _rotvec_dot_to_angular_rate_matrix(rotvecs):
    """Compute matrices to transform rot. vector derivatives to angular rates.

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
    k1 = np.empty_like(norm)
    k2 = np.empty_like(norm)
    mask = norm > 0.0001
    nm = norm[mask]
    k1[mask] = (1 - np.cos(nm)) / nm ** 2
    k2[mask] = (nm - np.sin(nm)) / nm ** 3
    mask = ~mask
    nm = norm[mask]
    k1[mask] = 0.5 - nm ** 2 / 24
    k2[mask] = 1 / 6 - nm ** 2 / 120
    skew = _create_skew_matrix(rotvecs)
    result = np.empty((len(rotvecs), 3, 3))
    result[:] = np.identity(3)
    result[:] -= k1[:, None, None] * skew
    result[:] += k2[:, None, None] * np.matmul(skew, skew)
    return result