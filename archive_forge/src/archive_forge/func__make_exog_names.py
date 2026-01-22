from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
def _make_exog_names(exog):
    exog_var = exog.var(0)
    if (exog_var == 0).any():
        const_idx = exog_var.argmin()
        exog_names = ['x%d' % i for i in range(1, exog.shape[1])]
        exog_names.insert(const_idx, 'const')
    else:
        exog_names = ['x%d' % i for i in range(1, exog.shape[1] + 1)]
    return exog_names