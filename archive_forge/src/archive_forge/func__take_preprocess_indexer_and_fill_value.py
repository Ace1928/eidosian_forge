from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import na_value_for_dtype
from pandas.core.construction import ensure_wrapped_if_datetimelike
def _take_preprocess_indexer_and_fill_value(arr: np.ndarray, indexer: npt.NDArray[np.intp], fill_value, allow_fill: bool, mask: npt.NDArray[np.bool_] | None=None):
    mask_info: tuple[np.ndarray | None, bool] | None = None
    if not allow_fill:
        dtype, fill_value = (arr.dtype, arr.dtype.type())
        mask_info = (None, False)
    else:
        dtype, fill_value = maybe_promote(arr.dtype, fill_value)
        if dtype != arr.dtype:
            if mask is not None:
                needs_masking = True
            else:
                mask = indexer == -1
                needs_masking = bool(mask.any())
            mask_info = (mask, needs_masking)
            if not needs_masking:
                dtype, fill_value = (arr.dtype, arr.dtype.type())
    return (dtype, fill_value, mask_info)