import numpy as np
from matplotlib import _api
def _view_transformation_uvw(u, v, w, E):
    """
    Return the view transformation matrix.

    Parameters
    ----------
    u : 3-element numpy array
        Unit vector pointing towards the right of the screen.
    v : 3-element numpy array
        Unit vector pointing towards the top of the screen.
    w : 3-element numpy array
        Unit vector pointing out of the screen.
    E : 3-element numpy array
        The coordinates of the eye/camera.
    """
    Mr = np.eye(4)
    Mt = np.eye(4)
    Mr[:3, :3] = [u, v, w]
    Mt[:3, -1] = -E
    M = np.dot(Mr, Mt)
    return M