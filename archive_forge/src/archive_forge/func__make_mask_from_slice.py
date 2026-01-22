from __future__ import annotations
from collections.abc import Iterable
from typing import (
import numpy as np
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
def _make_mask_from_slice(self, arg: slice) -> bool | np.ndarray:
    start = arg.start
    stop = arg.stop
    step = arg.step
    if step is not None and step < 0:
        raise ValueError(f'Invalid step {step}. Must be non-negative')
    mask: bool | np.ndarray = True
    if step is None:
        step = 1
    if start is None:
        if step > 1:
            mask &= self._ascending_count % step == 0
    elif start >= 0:
        mask &= self._ascending_count >= start
        if step > 1:
            mask &= (self._ascending_count - start) % step == 0
    else:
        mask &= self._descending_count < -start
        offset_array = self._descending_count + start + 1
        limit_array = self._ascending_count + self._descending_count + (start + 1) < 0
        offset_array = np.where(limit_array, self._ascending_count, offset_array)
        mask &= offset_array % step == 0
    if stop is not None:
        if stop >= 0:
            mask &= self._ascending_count < stop
        else:
            mask &= self._descending_count >= -stop
    return mask