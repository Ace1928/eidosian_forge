import os
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from .. import draw
from skimage import morphology
def _cross_decomposition(footprint, dtype=np.uint8):
    """Decompose a symmetric convex footprint into cross-shaped elements.

    This is a decomposition of the footprint into a sequence of
    (possibly asymmetric) cross-shaped elements. This technique was proposed in
    [1]_ and corresponds roughly to algorithm 1 of that publication (some
    details had to be modified to get reliable operation).

    .. [1] Li, D. and Ritter, G.X. Decomposition of Separable and Symmetric
           Convex Templates. Proc. SPIE 1350, Image Algebra and Morphological
           Image Processing, (1 November 1990).
           :DOI:`10.1117/12.23608`
    """
    quadrant = footprint[footprint.shape[0] // 2:, footprint.shape[1] // 2:]
    col_sums = quadrant.sum(0, dtype=int)
    col_sums = np.concatenate((col_sums, np.asarray([0], dtype=int)))
    i_prev = 0
    idx = {}
    sum0 = 0
    for i in range(col_sums.size - 1):
        if col_sums[i] > col_sums[i + 1]:
            if i == 0:
                continue
            key = (col_sums[i_prev] - col_sums[i], i - i_prev)
            sum0 += key[0]
            if key not in idx:
                idx[key] = 1
            else:
                idx[key] += 1
            i_prev = i
    n = quadrant.shape[0] - 1 - sum0
    if n > 0:
        key = (n, 0)
        idx[key] = idx.get(key, 0) + 1
    return tuple([(_cross(r0, r1, dtype), n) for (r0, r1), n in idx.items()])