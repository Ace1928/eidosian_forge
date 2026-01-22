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
def _ma_invtransparams(macoefs):
    """
    Inverse of the Jones reparameterization

    Parameters
    ----------
    params : ndarray
        The transformed MA coefficients
    """
    tmp = macoefs.copy()
    for j in range(len(macoefs) - 1, 0, -1):
        b = macoefs[j]
        for kiter in range(j):
            tmp[kiter] = (macoefs[kiter] - b * macoefs[j - kiter - 1]) / (1 - b ** 2)
        macoefs[:j] = tmp[:j]
    invmacoefs = -np.log((1 - macoefs) / (1 + macoefs))
    return invmacoefs