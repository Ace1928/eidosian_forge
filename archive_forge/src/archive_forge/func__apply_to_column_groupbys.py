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
def _apply_to_column_groupbys(self, func) -> DataFrame:
    from pandas.core.reshape.concat import concat
    obj = self._obj_with_exclusions
    columns = obj.columns
    sgbs = [SeriesGroupBy(obj.iloc[:, i], selection=colname, grouper=self._grouper, exclusions=self.exclusions, observed=self.observed) for i, colname in enumerate(obj.columns)]
    results = [func(sgb) for sgb in sgbs]
    if not len(results):
        res_df = DataFrame([], columns=columns, index=self._grouper.result_index)
    else:
        res_df = concat(results, keys=columns, axis=1)
    if not self.as_index:
        res_df.index = default_index(len(res_df))
        res_df = self._insert_inaxis_grouper(res_df)
    return res_df