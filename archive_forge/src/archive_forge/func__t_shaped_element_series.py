import os
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from .. import draw
from skimage import morphology
def _t_shaped_element_series(ndim=2, dtype=np.uint8):
    """A series of T-shaped structuring elements.

    In the 2D case this is a T-shaped element and its rotation at multiples of
    90 degrees. This series is used in efficient decompositions of disks of
    various radius as published in [1]_.

    The generalization to the n-dimensional case can be performed by having the
    "top" of the T to extend in (ndim - 1) dimensions and then producing a
    series of rotations such that the bottom end of the T points along each of
    ``2 * ndim`` orthogonal directions.
    """
    if ndim == 2:
        t0 = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=dtype)
        t90 = np.rot90(t0, 1)
        t180 = np.rot90(t0, 2)
        t270 = np.rot90(t0, 3)
        return (t0, t90, t180, t270)
    else:
        all_t = []
        for ax in range(ndim):
            for idx in [0, 2]:
                t = np.zeros((3,) * ndim, dtype=dtype)
                sl = [slice(None)] * ndim
                sl[ax] = slice(idx, idx + 1)
                t[tuple(sl)] = 1
                sl = [slice(1, 2)] * ndim
                sl[ax] = slice(None)
                t[tuple(sl)] = 1
                all_t.append(t)
    return tuple(all_t)