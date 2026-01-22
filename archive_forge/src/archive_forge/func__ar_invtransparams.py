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
def _ar_invtransparams(params):
    """
    Inverse of the Jones reparameterization

    Parameters
    ----------
    params : array_like
        The transformed AR coefficients
    """
    params = params.copy()
    tmp = params.copy()
    for j in range(len(params) - 1, 0, -1):
        a = params[j]
        for kiter in range(j):
            tmp[kiter] = (params[kiter] + a * params[j - kiter - 1]) / (1 - a ** 2)
        params[:j] = tmp[:j]
    invarcoefs = 2 * np.arctanh(params)
    return invarcoefs