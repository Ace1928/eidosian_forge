from __future__ import annotations
from functools import wraps
from typing import (
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
def _interpolate_with_limit_area(values: np.ndarray, method: Literal['pad', 'backfill'], limit: int | None, limit_area: Literal['inside', 'outside']) -> None:
    """
    Apply interpolation and limit_area logic to values along a to-be-specified axis.

    Parameters
    ----------
    values: np.ndarray
        Input array.
    method: str
        Interpolation method. Could be "bfill" or "pad"
    limit: int, optional
        Index limit on interpolation.
    limit_area: {'inside', 'outside'}
        Limit area for interpolation.

    Notes
    -----
    Modifies values in-place.
    """
    invalid = isna(values)
    is_valid = ~invalid
    if not invalid.all():
        first = find_valid_index(how='first', is_valid=is_valid)
        if first is None:
            first = 0
        last = find_valid_index(how='last', is_valid=is_valid)
        if last is None:
            last = len(values)
        pad_or_backfill_inplace(values, method=method, limit=limit, limit_area=limit_area)
        if limit_area == 'inside':
            invalid[first:last + 1] = False
        elif limit_area == 'outside':
            invalid[:first] = invalid[last + 1:] = False
        else:
            raise ValueError("limit_area should be 'inside' or 'outside'")
        values[invalid] = np.nan