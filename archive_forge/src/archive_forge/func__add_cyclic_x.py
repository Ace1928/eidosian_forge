import numpy as np
import numpy.ma as ma
def _add_cyclic_x(x, axis=-1, cyclic=360):
    """
    Add a cyclic point to a x/longitude coordinate array.

    Parameters
    ----------
    x : ndarray
        An array which specifies the x-coordinate values for
        the dimension the cyclic point is to be added to.
    axis : int, optional
        Specifies the axis of the x-coordinate array to add the cyclic point
        to. Defaults to the right-most axis.
    cyclic : float, optional
        Width of periodic domain (default: 360)

    Returns
    -------
    The coordinate array ``x`` with a cyclic point added.

    """
    npc = np.ma if np.ma.is_masked(x) else np
    cx = np.take(x, [0], axis=axis) + cyclic * np.sign(np.diff(np.take(x, [0, -1], axis=axis), axis=axis))
    return npc.concatenate((x, cx), axis=axis)