from __future__ import annotations
from collections.abc import (
import datetime
from functools import (
import inspect
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config.config import option_context
from pandas._libs import (
from pandas._libs.algos import rank_1d
import pandas._libs.groupby as libgroupby
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core._numba import executor
from pandas.core.apply import warn_alias_replacement
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
from pandas.core.arrays.string_arrow import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import (
from pandas.core.groupby.grouper import get_grouper
from pandas.core.groupby.indexing import (
from pandas.core.indexes.api import (
from pandas.core.internals.blocks import ensure_block_shape
from pandas.core.series import Series
from pandas.core.sorting import get_group_index_sorter
from pandas.core.util.numba_ import (
@final
def _concat_objects(self, values, not_indexed_same: bool=False, is_transform: bool=False):
    from pandas.core.reshape.concat import concat
    if self.group_keys and (not is_transform):
        if self.as_index:
            group_keys = self._grouper.result_index
            group_levels = self._grouper.levels
            group_names = self._grouper.names
            result = concat(values, axis=self.axis, keys=group_keys, levels=group_levels, names=group_names, sort=False)
        else:
            keys = list(range(len(values)))
            result = concat(values, axis=self.axis, keys=keys)
    elif not not_indexed_same:
        result = concat(values, axis=self.axis)
        ax = self._selected_obj._get_axis(self.axis)
        if self.dropna:
            labels = self._grouper.group_info[0]
            mask = labels != -1
            ax = ax[mask]
        if ax.has_duplicates and (not result.axes[self.axis].equals(ax)):
            target = algorithms.unique1d(ax._values)
            indexer, _ = result.index.get_indexer_non_unique(target)
            result = result.take(indexer, axis=self.axis)
        else:
            result = result.reindex(ax, axis=self.axis, copy=False)
    else:
        result = concat(values, axis=self.axis)
    if self.obj.ndim == 1:
        name = self.obj.name
    elif is_hashable(self._selection):
        name = self._selection
    else:
        name = None
    if isinstance(result, Series) and name is not None:
        result.name = name
    return result