import numpy as np
from scipy.linalg import solve_banded
from ._rotation import Rotation
def _angular_acceleration_nonlinear_term(rotvecs, rotvecs_dot):
    """Compute the non-linear term in angular acceleration.

    The angular acceleration contains a quadratic term with respect to
    the derivative of the rotation vector. This function computes that.

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
    norm = np.linalg.norm(rotvecs, axis=1)
    dp = np.sum(rotvecs * rotvecs_dot, axis=1)
    cp = np.cross(rotvecs, rotvecs_dot)
    ccp = np.cross(rotvecs, cp)
    dccp = np.cross(rotvecs_dot, cp)
    k1 = np.empty_like(norm)
    k2 = np.empty_like(norm)
    k3 = np.empty_like(norm)
    mask = norm > 0.0001
    nm = norm[mask]
    k1[mask] = (-nm * np.sin(nm) - 2 * (np.cos(nm) - 1)) / nm ** 4
    k2[mask] = (-2 * nm + 3 * np.sin(nm) - nm * np.cos(nm)) / nm ** 5
    k3[mask] = (nm - np.sin(nm)) / nm ** 3
    mask = ~mask
    nm = norm[mask]
    k1[mask] = 1 / 12 - nm ** 2 / 180
    k2[mask] = -1 / 60 + nm ** 2 / 12604
    k3[mask] = 1 / 6 - nm ** 2 / 120
    dp = dp[:, None]
    k1 = k1[:, None]
    k2 = k2[:, None]
    k3 = k3[:, None]
    return dp * (k1 * cp + k2 * ccp) + k3 * dccp