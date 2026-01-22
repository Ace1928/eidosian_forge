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
def _reindex_output(self, output: OutputFrameOrSeries, fill_value: Scalar=np.nan, qs: npt.NDArray[np.float64] | None=None) -> OutputFrameOrSeries:
    """
        If we have categorical groupers, then we might want to make sure that
        we have a fully re-indexed output to the levels. This means expanding
        the output space to accommodate all values in the cartesian product of
        our groups, regardless of whether they were observed in the data or
        not. This will expand the output space if there are missing groups.

        The method returns early without modifying the input if the number of
        groupings is less than 2, self.observed == True or none of the groupers
        are categorical.

        Parameters
        ----------
        output : Series or DataFrame
            Object resulting from grouping and applying an operation.
        fill_value : scalar, default np.nan
            Value to use for unobserved categories if self.observed is False.
        qs : np.ndarray[float64] or None, default None
            quantile values, only relevant for quantile.

        Returns
        -------
        Series or DataFrame
            Object (potentially) re-indexed to include all possible groups.
        """
    groupings = self._grouper.groupings
    if len(groupings) == 1:
        return output
    elif self.observed:
        return output
    elif not any((isinstance(ping.grouping_vector, (Categorical, CategoricalIndex)) for ping in groupings)):
        return output
    levels_list = [ping._group_index for ping in groupings]
    names = self._grouper.names
    if qs is not None:
        levels_list.append(qs)
        names = names + [None]
    index = MultiIndex.from_product(levels_list, names=names)
    if self.sort:
        index = index.sort_values()
    if self.as_index:
        d = {self.obj._get_axis_name(self.axis): index, 'copy': False, 'fill_value': fill_value}
        return output.reindex(**d)
    in_axis_grps = [(i, ping.name) for i, ping in enumerate(groupings) if ping.in_axis]
    if len(in_axis_grps) > 0:
        g_nums, g_names = zip(*in_axis_grps)
        output = output.drop(labels=list(g_names), axis=1)
    output = output.set_index(self._grouper.result_index).reindex(index, copy=False, fill_value=fill_value)
    if len(in_axis_grps) > 0:
        output = output.reset_index(level=g_nums)
    return output.reset_index(drop=True)