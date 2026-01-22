import numpy as np
from . import check_consistent_length, check_matplotlib_support
from ._response import _get_response_values_binary
from .multiclass import type_of_target
from .validation import _check_pos_label_consistency
def _interval_max_min_ratio(data):
    """Compute the ratio between the largest and smallest inter-point distances.

    A value larger than 5 typically indicates that the parameter range would
    better be displayed with a log scale while a linear scale would be more
    suitable otherwise.
    """
    diff = np.diff(np.sort(data))
    return diff.max() / diff.min()