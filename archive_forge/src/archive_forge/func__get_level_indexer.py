from __future__ import annotations
from collections.abc import (
from functools import wraps
from sys import getsizeof
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import (
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.construction import sanitize_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.io.formats.printing import (
def _get_level_indexer(self, key, level: int=0, indexer: npt.NDArray[np.bool_] | None=None):
    level_index = self.levels[level]
    level_codes = self.codes[level]

    def convert_indexer(start, stop, step, indexer=indexer, codes=level_codes):
        if indexer is not None:
            codes = codes[indexer]
        if step is None or step == 1:
            new_indexer = (codes >= start) & (codes < stop)
        else:
            r = np.arange(start, stop, step, dtype=codes.dtype)
            new_indexer = algos.isin(codes, r)
        if indexer is None:
            return new_indexer
        indexer = indexer.copy()
        indexer[indexer] = new_indexer
        return indexer
    if isinstance(key, slice):
        step = key.step
        is_negative_step = step is not None and step < 0
        try:
            if key.start is not None:
                start = level_index.get_loc(key.start)
            elif is_negative_step:
                start = len(level_index) - 1
            else:
                start = 0
            if key.stop is not None:
                stop = level_index.get_loc(key.stop)
            elif is_negative_step:
                stop = 0
            elif isinstance(start, slice):
                stop = len(level_index)
            else:
                stop = len(level_index) - 1
        except KeyError:
            start = stop = level_index.slice_indexer(key.start, key.stop, key.step)
            step = start.step
        if isinstance(start, slice) or isinstance(stop, slice):
            start = getattr(start, 'start', start)
            stop = getattr(stop, 'stop', stop)
            return convert_indexer(start, stop, step)
        elif level > 0 or self._lexsort_depth == 0 or step is not None:
            stop = stop - 1 if is_negative_step else stop + 1
            return convert_indexer(start, stop, step)
        else:
            i = algos.searchsorted(level_codes, start, side='left')
            j = algos.searchsorted(level_codes, stop, side='right')
            return slice(i, j, step)
    else:
        idx = self._get_loc_single_level_index(level_index, key)
        if level > 0 or self._lexsort_depth == 0:
            if isinstance(idx, slice):
                locs = (level_codes >= idx.start) & (level_codes < idx.stop)
                return locs
            locs = np.array(level_codes == idx, dtype=bool, copy=False)
            if not locs.any():
                raise KeyError(key)
            return locs
        if isinstance(idx, slice):
            start = algos.searchsorted(level_codes, idx.start, side='left')
            end = algos.searchsorted(level_codes, idx.stop, side='left')
        else:
            start = algos.searchsorted(level_codes, idx, side='left')
            end = algos.searchsorted(level_codes, idx, side='right')
        if start == end:
            raise KeyError(key)
        return slice(start, end)