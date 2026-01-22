from __future__ import annotations
import math
import numpy as np
from scipy import special
from ._axis_nan_policy import _axis_nan_policy_factory, _broadcast_arrays
def _correa_entropy(X, m):
    """Compute the Correa estimator as described in [6]."""
    n = X.shape[-1]
    X = _pad_along_last_axis(X, m)
    i = np.arange(1, n + 1)
    dj = np.arange(-m, m + 1)[:, None]
    j = i + dj
    j0 = j + m - 1
    Xibar = np.mean(X[..., j0], axis=-2, keepdims=True)
    difference = X[..., j0] - Xibar
    num = np.sum(difference * dj, axis=-2)
    den = n * np.sum(difference ** 2, axis=-2)
    return -np.mean(np.log(num / den), axis=-1)