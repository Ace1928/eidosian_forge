import math
from functools import reduce
import numpy as np
def angle_axis2euler(theta, vector, is_normalized=False):
    """Convert angle, axis pair to Euler angles

    Parameters
    ----------
    theta : scalar
       angle of rotation
    vector : 3 element sequence
       vector specifying axis for rotation.
    is_normalized : bool, optional
       True if vector is already normalized (has norm of 1).  Default
       False

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Examples
    --------
    >>> z, y, x = angle_axis2euler(0, [1, 0, 0])
    >>> np.allclose((z, y, x), 0)
    True

    Notes
    -----
    It's possible to reduce the amount of calculation a little, by
    combining parts of the ``angle_axis2mat`` and ``mat2euler``
    functions, but the reduction in computation is small, and the code
    repetition is large.
    """
    from . import quaternions as nq
    M = nq.angle_axis2mat(theta, vector, is_normalized)
    return mat2euler(M)