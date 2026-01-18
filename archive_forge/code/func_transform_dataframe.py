from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def transform_dataframe(self, dataframe, function, level=0, **kwargs):
    """Apply function to each column, by group
        Assumes that the dataframe already has a proper index"""
    if dataframe.shape[0] != self.nobs:
        raise Exception('dataframe does not have the same shape as index')
    out = dataframe.groupby(level=level).apply(function, **kwargs)
    if 1 in out.shape:
        return np.ravel(out)
    else:
        return np.array(out)