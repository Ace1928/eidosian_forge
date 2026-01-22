from __future__ import annotations
import math
import numpy as np
from scipy import special
from ._axis_nan_policy import _axis_nan_policy_factory, _broadcast_arrays
def _pad_along_last_axis(X, m):
    """Pad the data for computing the rolling window difference."""
    shape = np.array(X.shape)
    shape[-1] = m
    Xl = np.broadcast_to(X[..., [0]], shape)
    Xr = np.broadcast_to(X[..., [-1]], shape)
    return np.concatenate((Xl, X, Xr), axis=-1)