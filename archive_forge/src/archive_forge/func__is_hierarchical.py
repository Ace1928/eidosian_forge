from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def _is_hierarchical(x):
    """
    Checks if the first item of an array-like object is also array-like
    If so, we have a MultiIndex and returns True. Else returns False.
    """
    item = x[0]
    if isinstance(item, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)):
        return True
    else:
        return False