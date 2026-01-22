import numpy as np
from collections import namedtuple
@wrap(no_cpython_wrapper=True)
def argmergesort(arr):
    """Out-of-place"""
    idxs = np.arange(arr.size)
    ws = np.empty(arr.size // 2, dtype=idxs.dtype)
    argmergesort_inner(idxs, arr, ws)
    return idxs