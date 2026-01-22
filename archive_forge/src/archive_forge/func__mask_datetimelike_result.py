from __future__ import annotations
import functools
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
def _mask_datetimelike_result(result: np.ndarray | np.datetime64 | np.timedelta64, axis: AxisInt | None, mask: npt.NDArray[np.bool_], orig_values: np.ndarray) -> np.ndarray | np.datetime64 | np.timedelta64 | NaTType:
    if isinstance(result, np.ndarray):
        result = result.astype('i8').view(orig_values.dtype)
        axis_mask = mask.any(axis=axis)
        result[axis_mask] = iNaT
    elif mask.any():
        return np.int64(iNaT).view(orig_values.dtype)
    return result