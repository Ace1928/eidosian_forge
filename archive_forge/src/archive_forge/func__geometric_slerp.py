from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from scipy.spatial.distance import euclidean
def _geometric_slerp(start, end, t):
    basis = np.vstack([start, end])
    Q, R = np.linalg.qr(basis.T)
    signs = 2 * (np.diag(R) >= 0) - 1
    Q = Q.T * signs.T[:, np.newaxis]
    R = R.T * signs.T[:, np.newaxis]
    c = np.dot(start, end)
    s = np.linalg.det(R)
    omega = np.arctan2(s, c)
    start, end = Q
    s = np.sin(t * omega)
    c = np.cos(t * omega)
    return start * c[:, np.newaxis] + end * s[:, np.newaxis]