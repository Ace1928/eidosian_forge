from __future__ import annotations
import math
import numpy as np
from scipy import special
from ._axis_nan_policy import _axis_nan_policy_factory, _broadcast_arrays
def _van_es_entropy(X, m):
    """Compute the van Es estimator as described in [6]."""
    n = X.shape[-1]
    difference = X[..., m:] - X[..., :-m]
    term1 = 1 / (n - m) * np.sum(np.log((n + 1) / m * difference), axis=-1)
    k = np.arange(m, n + 1)
    return term1 + np.sum(1 / k) + np.log(m) - np.log(n + 1)