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
@_datetimelike_compat
def _backfill_1d(values: np.ndarray, limit: int | None=None, limit_area: Literal['inside', 'outside'] | None=None, mask: npt.NDArray[np.bool_] | None=None) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    mask = _fillna_prep(values, mask)
    if limit_area is not None and (not mask.all()):
        _fill_limit_area_1d(mask, limit_area)
    algos.backfill_inplace(values, mask, limit=limit)
    return (values, mask)