import numbers
import warnings
from collections import Counter
from functools import partial
import numpy as np
import numpy.ma as ma
from scipy import sparse as sp
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import _is_pandas_na, is_scalar_nan
from ..utils._mask import _get_mask
from ..utils._param_validation import MissingValues, StrOptions
from ..utils.fixes import _mode
from ..utils.sparsefuncs import _get_median
from ..utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted
def _most_frequent(array, extra_value, n_repeat):
    """Compute the most frequent value in a 1d array extended with
    [extra_value] * n_repeat, where extra_value is assumed to be not part
    of the array."""
    if array.size > 0:
        if array.dtype == object:
            counter = Counter(array)
            most_frequent_count = counter.most_common(1)[0][1]
            most_frequent_value = min((value for value, count in counter.items() if count == most_frequent_count))
        else:
            mode = _mode(array)
            most_frequent_value = mode[0][0]
            most_frequent_count = mode[1][0]
    else:
        most_frequent_value = 0
        most_frequent_count = 0
    if most_frequent_count == 0 and n_repeat == 0:
        return np.nan
    elif most_frequent_count < n_repeat:
        return extra_value
    elif most_frequent_count > n_repeat:
        return most_frequent_value
    elif most_frequent_count == n_repeat:
        return min(most_frequent_value, extra_value)