import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
def _apply_mat(self, coords, matrix):
    ndim = matrix.shape[0] - 1
    coords = np.array(coords, copy=NP_COPY_IF_NEEDED, ndmin=2)
    src = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
    dst = src @ matrix.T
    dst[dst[:, ndim] == 0, ndim] = np.finfo(float).eps
    dst[:, :ndim] /= dst[:, ndim:ndim + 1]
    return dst[:, :ndim]