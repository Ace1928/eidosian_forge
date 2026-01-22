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
def _get_splitter(self, data: NDFrame, axis: AxisInt=0) -> DataSplitter:
    """
        Returns
        -------
        Generator yielding subsetted objects
        """
    ids, _, ngroups = self.group_info
    return _get_splitter(data, ids, ngroups, sorted_ids=self._sorted_ids, sort_idx=self._sort_idx, axis=axis)