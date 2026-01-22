from __future__ import annotations
from collections import abc
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.indexes.api import (
from pandas.core.internals import concatenate_managers
def _get_sample_object(self, objs: list[Series | DataFrame], ndims: set[int], keys, names, levels) -> tuple[Series | DataFrame, list[Series | DataFrame]]:
    sample: Series | DataFrame | None = None
    if len(ndims) > 1:
        max_ndim = max(ndims)
        for obj in objs:
            if obj.ndim == max_ndim and np.sum(obj.shape):
                sample = obj
                break
    else:
        non_empties = [obj for obj in objs if sum(obj.shape) > 0 or obj.ndim == 1]
        if len(non_empties) and (keys is None and names is None and (levels is None) and (not self.intersect)):
            objs = non_empties
            sample = objs[0]
    if sample is None:
        sample = objs[0]
    return (sample, objs)