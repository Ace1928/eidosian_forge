from functools import reduce
import numpy as np
from ..draw import polygon
from .._shared.version_requirements import require
def _mask_from_vertices(vertices, shape, label):
    mask = np.zeros(shape, dtype=int)
    pr = [y for x, y in vertices]
    pc = [x for x, y in vertices]
    rr, cc = polygon(pr, pc, shape)
    mask[rr, cc] = label
    return mask