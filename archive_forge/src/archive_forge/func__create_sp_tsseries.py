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
def _create_sp_tsseries():
    nan = np.nan
    arr = np.arange(15, dtype=np.float64)
    arr[7:12] = nan
    arr[-1:] = nan
    date_index = bdate_range('1/1/2011', periods=len(arr))
    bseries = Series(SparseArray(arr, kind='block'), index=date_index)
    bseries.name = 'btsseries'
    return bseries