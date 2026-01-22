from datetime import timedelta
import os
import pickle
import platform as pl
import sys
import numpy as np
import pandas
from pandas import (
from pandas.arrays import SparseArray
from pandas.tseries.offsets import (
def _create_sp_frame():
    nan = np.nan
    data = {'A': [nan, nan, nan, 0, 1, 2, 3, 4, 5, 6], 'B': [0, 1, 2, nan, nan, nan, 3, 4, 5, 6], 'C': np.arange(10).astype(np.int64), 'D': [0, 1, 2, 3, 4, 5, nan, nan, nan, nan]}
    dates = bdate_range('1/1/2011', periods=10)
    return DataFrame(data, index=dates).apply(SparseArray)