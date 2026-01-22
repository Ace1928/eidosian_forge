import numpy as np
import numba
from .exceptions import ParameterError
from .utils import valid_intervals
from .._typing import _SequenceLike
@numba.jit(nopython=True, cache=True)
def __match_events_helper(output: np.ndarray, events_from: np.ndarray, events_to: np.ndarray, left: bool=True, right: bool=True):
    from_idx = np.argsort(events_from)
    sorted_from = events_from[from_idx]
    to_idx = np.argsort(events_to)
    sorted_to = events_to[to_idx]
    matching_indices = np.searchsorted(sorted_to, sorted_from)
    for ind, middle_ind in enumerate(matching_indices):
        left_flag = False
        right_flag = False
        left_ind = -1
        right_ind = len(matching_indices)
        left_diff = 0
        right_diff = 0
        mid_diff = 0
        middle_ind = matching_indices[ind]
        sorted_from_num = sorted_from[ind]
        if middle_ind == len(sorted_to):
            middle_ind -= 1
        if left and middle_ind > 0:
            left_ind = middle_ind - 1
            left_flag = True
        if right and middle_ind < len(sorted_to) - 1:
            right_ind = middle_ind + 1
            right_flag = True
        mid_diff = abs(sorted_to[middle_ind] - sorted_from_num)
        if left and left_flag:
            left_diff = abs(sorted_to[left_ind] - sorted_from_num)
        if right and right_flag:
            right_diff = abs(sorted_to[right_ind] - sorted_from_num)
        if left_flag and (not right and sorted_to[middle_ind] > sorted_from_num or (not right_flag and left_diff < mid_diff) or (left_diff < right_diff and left_diff < mid_diff)):
            output[ind] = to_idx[left_ind]
        elif right_flag and right_diff < mid_diff:
            output[ind] = to_idx[right_ind]
        else:
            output[ind] = to_idx[middle_ind]
    solutions = np.empty_like(output)
    solutions[from_idx] = output
    return solutions