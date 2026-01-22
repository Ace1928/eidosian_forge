from __future__ import annotations
from collections.abc import Sequence
from typing import (
import warnings
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
def _sanitize_ndim(result: ArrayLike, data, dtype: DtypeObj | None, index: Index | None, *, allow_2d: bool=False) -> ArrayLike:
    """
    Ensure we have a 1-dimensional result array.
    """
    if getattr(result, 'ndim', 0) == 0:
        raise ValueError('result should be arraylike with ndim > 0')
    if result.ndim == 1:
        result = _maybe_repeat(result, index)
    elif result.ndim > 1:
        if isinstance(data, np.ndarray):
            if allow_2d:
                return result
            raise ValueError(f'Data must be 1-dimensional, got ndarray of shape {data.shape} instead')
        if is_object_dtype(dtype) and isinstance(dtype, ExtensionDtype):
            result = com.asarray_tuplesafe(data, dtype=np.dtype('object'))
            cls = dtype.construct_array_type()
            result = cls._from_sequence(result, dtype=dtype)
        else:
            result = com.asarray_tuplesafe(data, dtype=dtype)
    return result