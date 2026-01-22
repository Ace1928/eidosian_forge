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
def _try_cast(arr: list | np.ndarray, dtype: np.dtype, copy: bool) -> ArrayLike:
    """
    Convert input to numpy ndarray and optionally cast to a given dtype.

    Parameters
    ----------
    arr : ndarray or list
        Excludes: ExtensionArray, Series, Index.
    dtype : np.dtype
    copy : bool
        If False, don't copy the data if not needed.

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    is_ndarray = isinstance(arr, np.ndarray)
    if dtype == object:
        if not is_ndarray:
            subarr = construct_1d_object_array_from_listlike(arr)
            return subarr
        return ensure_wrapped_if_datetimelike(arr).astype(dtype, copy=copy)
    elif dtype.kind == 'U':
        if is_ndarray:
            arr = cast(np.ndarray, arr)
            shape = arr.shape
            if arr.ndim > 1:
                arr = arr.ravel()
        else:
            shape = (len(arr),)
        return lib.ensure_string_array(arr, convert_na_value=False, copy=copy).reshape(shape)
    elif dtype.kind in 'mM':
        return maybe_cast_to_datetime(arr, dtype)
    elif dtype.kind in 'iu':
        subarr = maybe_cast_to_integer_array(arr, dtype)
    else:
        subarr = np.array(arr, dtype=dtype, copy=copy)
    return subarr