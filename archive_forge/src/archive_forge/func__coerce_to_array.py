from __future__ import annotations
import numbers
from typing import (
import numpy as np
from pandas._libs import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import (
from pandas.core.arrays.masked import (
@classmethod
def _coerce_to_array(cls, value, *, dtype: DtypeObj, copy: bool=False) -> tuple[np.ndarray, np.ndarray]:
    dtype_cls = cls._dtype_cls
    default_dtype = dtype_cls._default_np_dtype
    values, mask, _, _ = _coerce_to_data_and_mask(value, dtype, copy, dtype_cls, default_dtype)
    return (values, mask)