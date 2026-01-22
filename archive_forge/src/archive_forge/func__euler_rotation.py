import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
def _euler_rotation(axis, angle):
    """Produce a single-axis Euler rotation matrix.

    Parameters
    ----------
    axis : int in {0, 1, 2}
        The axis of rotation.
    angle : float
        The angle of rotation in radians.

    Returns
    -------
    Ri : array of float, shape (3, 3)
        The rotation matrix along axis `axis`.
    """
    i = axis
    s = (-1) ** i * np.sin(angle)
    c = np.cos(angle)
    R2 = np.array([[c, -s], [s, c]])
    Ri = np.eye(3)
    axes = sorted({0, 1, 2} - {axis})
    sl = slice(axes[0], axes[1] + 1, axes[1] - axes[0])
    Ri[sl, sl] = R2
    return Ri