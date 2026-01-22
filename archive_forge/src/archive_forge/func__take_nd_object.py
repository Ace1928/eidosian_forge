from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import na_value_for_dtype
from pandas.core.construction import ensure_wrapped_if_datetimelike
def _take_nd_object(arr: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, axis: AxisInt, fill_value, mask_info) -> None:
    if mask_info is not None:
        mask, needs_masking = mask_info
    else:
        mask = indexer == -1
        needs_masking = mask.any()
    if arr.dtype != out.dtype:
        arr = arr.astype(out.dtype)
    if arr.shape[axis] > 0:
        arr.take(indexer, axis=axis, out=out)
    if needs_masking:
        outindexer = [slice(None)] * arr.ndim
        outindexer[axis] = mask
        out[tuple(outindexer)] = fill_value