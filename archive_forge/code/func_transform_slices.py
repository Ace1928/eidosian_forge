from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def transform_slices(self, array, function, level=0, **kwargs):
    """Apply function to each group. Similar to transform_array but does
        not coerce array to a DataFrame and back and only works on a 1D or 2D
        numpy array. function is called function(group, group_idx, **kwargs).
        """
    array = np.asarray(array)
    if array.shape[0] != self.nobs:
        raise Exception('array does not have the same shape as index')
    self.get_slices(level=level)
    processed = []
    for s in self.slices:
        if array.ndim == 2:
            subset = array[s, :]
        elif array.ndim == 1:
            subset = array[s]
        processed.append(function(subset, s, **kwargs))
    processed = np.array(processed)
    return processed.reshape(-1, processed.shape[-1])