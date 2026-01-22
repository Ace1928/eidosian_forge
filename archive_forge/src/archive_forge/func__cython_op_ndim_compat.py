from __future__ import annotations
import collections
import functools
from typing import (
import numpy as np
from pandas._libs import (
import pandas._libs.groupby as libgroupby
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import (
@final
def _cython_op_ndim_compat(self, values: np.ndarray, *, min_count: int, ngroups: int, comp_ids: np.ndarray, mask: npt.NDArray[np.bool_] | None=None, result_mask: npt.NDArray[np.bool_] | None=None, **kwargs) -> np.ndarray:
    if values.ndim == 1:
        values2d = values[None, :]
        if mask is not None:
            mask = mask[None, :]
        if result_mask is not None:
            result_mask = result_mask[None, :]
        res = self._call_cython_op(values2d, min_count=min_count, ngroups=ngroups, comp_ids=comp_ids, mask=mask, result_mask=result_mask, **kwargs)
        if res.shape[0] == 1:
            return res[0]
        return res.T
    return self._call_cython_op(values, min_count=min_count, ngroups=ngroups, comp_ids=comp_ids, mask=mask, result_mask=result_mask, **kwargs)