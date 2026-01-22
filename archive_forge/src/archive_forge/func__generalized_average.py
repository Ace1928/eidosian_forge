import warnings
from math import log
from numbers import Real
import numpy as np
from scipy import sparse as sp
from ...utils._param_validation import Interval, StrOptions, validate_params
from ...utils.multiclass import type_of_target
from ...utils.validation import check_array, check_consistent_length
from ._expected_mutual_info_fast import expected_mutual_information
def _generalized_average(U, V, average_method):
    """Return a particular mean of two numbers."""
    if average_method == 'min':
        return min(U, V)
    elif average_method == 'geometric':
        return np.sqrt(U * V)
    elif average_method == 'arithmetic':
        return np.mean([U, V])
    elif average_method == 'max':
        return max(U, V)
    else:
        raise ValueError("'average_method' must be 'min', 'geometric', 'arithmetic', or 'max'")