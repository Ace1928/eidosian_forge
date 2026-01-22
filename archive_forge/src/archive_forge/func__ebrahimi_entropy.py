from __future__ import annotations
import math
import numpy as np
from scipy import special
from ._axis_nan_policy import _axis_nan_policy_factory, _broadcast_arrays
def _ebrahimi_entropy(X, m):
    """Compute the Ebrahimi estimator as described in [6]."""
    n = X.shape[-1]
    X = _pad_along_last_axis(X, m)
    differences = X[..., 2 * m:] - X[..., :-2 * m]
    i = np.arange(1, n + 1).astype(float)
    ci = np.ones_like(i) * 2
    ci[i <= m] = 1 + (i[i <= m] - 1) / m
    ci[i >= n - m + 1] = 1 + (n - i[i >= n - m + 1]) / m
    logs = np.log(n * differences / (ci * m))
    return np.mean(logs, axis=-1)