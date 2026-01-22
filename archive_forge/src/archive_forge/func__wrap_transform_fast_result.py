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
def _wrap_transform_fast_result(self, result: NDFrameT) -> NDFrameT:
    """
        Fast transform path for aggregations.
        """
    obj = self._obj_with_exclusions
    ids, _, _ = self._grouper.group_info
    result = result.reindex(self._grouper.result_index, axis=self.axis, copy=False)
    if self.obj.ndim == 1:
        out = algorithms.take_nd(result._values, ids)
        output = obj._constructor(out, index=obj.index, name=obj.name)
    else:
        axis = 0 if result.ndim == 1 else self.axis
        new_ax = result.axes[axis].take(ids)
        output = result._reindex_with_indexers({axis: (new_ax, ids)}, allow_dups=True, copy=False)
        output = output.set_axis(obj._get_axis(self.axis), axis=axis)
    return output