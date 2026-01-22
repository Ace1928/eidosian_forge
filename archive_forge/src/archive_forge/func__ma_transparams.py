from __future__ import annotations
from statsmodels.compat.python import lrange
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import Literal
from statsmodels.tools.data import _is_recarray, _is_using_pandas
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.typing import NDArray
from statsmodels.tools.validation import (
def _ma_transparams(params):
    """
    Transforms params to induce stationarity/invertability.

    Parameters
    ----------
    params : ndarray
        The ma coeffecients of an (AR)MA model.

    Reference
    ---------
    Jones(1980)
    """
    newparams = ((1 - np.exp(-params)) / (1 + np.exp(-params))).copy()
    tmp = ((1 - np.exp(-params)) / (1 + np.exp(-params))).copy()
    for j in range(1, len(params)):
        b = newparams[j]
        for kiter in range(j):
            tmp[kiter] += b * newparams[j - kiter - 1]
        newparams[:j] = tmp[:j]
    return newparams