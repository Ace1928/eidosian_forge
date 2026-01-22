from __future__ import annotations
from collections import abc
from functools import partial
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas.errors import SpecificationError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import (
from pandas.core import algorithms
from pandas.core.apply import (
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import (
from pandas.core.groupby.groupby import (
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import get_group_index
from pandas.core.util.numba_ import maybe_use_numba
from pandas.plotting import boxplot_frame_groupby
def _wrap_applied_output_series(self, values: list[Series], not_indexed_same: bool, first_not_none, key_index: Index | None, is_transform: bool) -> DataFrame | Series:
    kwargs = first_not_none._construct_axes_dict()
    backup = Series(**kwargs)
    values = [x if x is not None else backup for x in values]
    all_indexed_same = all_indexes_same((x.index for x in values))
    if not all_indexed_same:
        return self._concat_objects(values, not_indexed_same=True, is_transform=is_transform)
    stacked_values = np.vstack([np.asarray(v) for v in values])
    if self.axis == 0:
        index = key_index
        columns = first_not_none.index.copy()
        if columns.name is None:
            names = {v.name for v in values}
            if len(names) == 1:
                columns.name = next(iter(names))
    else:
        index = first_not_none.index
        columns = key_index
        stacked_values = stacked_values.T
    if stacked_values.dtype == object:
        stacked_values = stacked_values.tolist()
    result = self.obj._constructor(stacked_values, index=index, columns=columns)
    if not self.as_index:
        result = self._insert_inaxis_grouper(result)
    return self._reindex_output(result)