from collections import Counter
from contextlib import suppress
from typing import NamedTuple
import numpy as np
from . import is_scalar_nan
def _get_counts(values, uniques):
    """Get the count of each of the `uniques` in `values`.

    The counts will use the order passed in by `uniques`. For non-object dtypes,
    `uniques` is assumed to be sorted and `np.nan` is at the end.
    """
    if values.dtype.kind in 'OU':
        counter = _NaNCounter(values)
        output = np.zeros(len(uniques), dtype=np.int64)
        for i, item in enumerate(uniques):
            with suppress(KeyError):
                output[i] = counter[item]
        return output
    unique_values, counts = _unique_np(values, return_counts=True)
    uniques_in_values = np.isin(uniques, unique_values, assume_unique=True)
    if np.isnan(unique_values[-1]) and np.isnan(uniques[-1]):
        uniques_in_values[-1] = True
    unique_valid_indices = np.searchsorted(unique_values, uniques[uniques_in_values])
    output = np.zeros_like(uniques, dtype=np.int64)
    output[uniques_in_values] = counts[unique_valid_indices]
    return output