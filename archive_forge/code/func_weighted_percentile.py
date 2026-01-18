import numpy as np
import pandas as pd
from .._utils import resolution
from ..doctools import document
from .stat import stat
def weighted_percentile(a, q, weights=None):
    """
    Compute the weighted q-th percentile of data

    Parameters
    ----------
    a : array_like
        Input that can be converted into an array.
    q : array_like[float]
        Percentile or sequence of percentiles to compute. Must be int
        the range [0, 100]
    weights : array_like
        Weights associated with the input values.
    """
    if weights is None:
        weights = np.ones(len(a))
    weights = np.asarray(weights)
    q = np.asarray(q)
    C = 1
    idx_s = np.argsort(a)
    a_s = a[idx_s]
    w_n = weights[idx_s]
    S_N = np.sum(weights)
    S_n = np.cumsum(w_n)
    p_n = (S_n - C * w_n) / (S_N + (1 - 2 * C) * w_n)
    pcts = np.interp(q / 100.0, p_n, a_s)
    return pcts