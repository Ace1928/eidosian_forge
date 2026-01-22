from __future__ import annotations
import inspect
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.errors import IntCastingNaNError
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
def _astype_float_to_int_nansafe(values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
    """
    astype with a check preventing converting NaN to an meaningless integer value.
    """
    if not np.isfinite(values).all():
        raise IntCastingNaNError('Cannot convert non-finite values (NA or inf) to integer')
    if dtype.kind == 'u':
        if not (values >= 0).all():
            raise ValueError(f'Cannot losslessly cast from {values.dtype} to {dtype}')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        return values.astype(dtype, copy=copy)