from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
@classmethod
def _drop_nans_2d(cls, x, nan_mask):
    if isinstance(x, (Series, DataFrame)):
        return x.loc[nan_mask].loc[:, nan_mask]
    else:
        return super()._drop_nans_2d(x, nan_mask)