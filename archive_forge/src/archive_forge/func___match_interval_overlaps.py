import numpy as np
import numba
from .exceptions import ParameterError
from .utils import valid_intervals
from .._typing import _SequenceLike
@numba.jit(nopython=True, cache=True)
def __match_interval_overlaps(query, intervals_to, candidates):
    """Find the best Jaccard match from query to candidates"""
    best_score = -1
    best_idx = -1
    for idx in candidates:
        score = __jaccard(query, intervals_to[idx])
        if score > best_score:
            best_score, best_idx = (score, idx)
    return best_idx