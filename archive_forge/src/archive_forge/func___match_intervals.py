import numpy as np
import numba
from .exceptions import ParameterError
from .utils import valid_intervals
from .._typing import _SequenceLike
@numba.jit(nopython=True, cache=True)
def __match_intervals(intervals_from: np.ndarray, intervals_to: np.ndarray, strict: bool=True) -> np.ndarray:
    """Numba-accelerated interval matching algorithm."""
    start_index = np.argsort(intervals_to[:, 0])
    end_index = np.argsort(intervals_to[:, 1])
    start_sorted = intervals_to[start_index, 0]
    end_sorted = intervals_to[end_index, 1]
    search_ends = np.searchsorted(start_sorted, intervals_from[:, 1], side='right')
    search_starts = np.searchsorted(end_sorted, intervals_from[:, 0], side='left')
    output = np.empty(len(intervals_from), dtype=numba.uint32)
    for i in range(len(intervals_from)):
        query = intervals_from[i]
        after_query = search_ends[i]
        before_query = search_starts[i]
        candidates = set(start_index[:after_query]) & set(end_index[before_query:])
        if len(candidates) > 0:
            output[i] = __match_interval_overlaps(query, intervals_to, candidates)
        elif strict:
            raise ParameterError
        else:
            dist_before = np.inf
            dist_after = np.inf
            if search_starts[i] > 0:
                dist_before = query[0] - end_sorted[search_starts[i] - 1]
            if search_ends[i] + 1 < len(intervals_to):
                dist_after = start_sorted[search_ends[i] + 1] - query[1]
            if dist_before < dist_after:
                output[i] = end_index[search_starts[i] - 1]
            else:
                output[i] = start_index[search_ends[i] + 1]
    return output