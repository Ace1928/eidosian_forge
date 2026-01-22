import numpy as np
from matplotlib import _api
def _vec_pad_ones(xs, ys, zs):
    return np.array([xs, ys, zs, np.ones_like(xs)])