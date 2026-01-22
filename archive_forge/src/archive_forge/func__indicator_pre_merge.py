from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from typing import (
import uuid
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas._typing import (
from pandas.errors import MergeError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.frame import _merge_doc
from pandas.core.indexes.api import default_index
from pandas.core.sorting import (
@final
def _indicator_pre_merge(self, left: DataFrame, right: DataFrame) -> tuple[DataFrame, DataFrame]:
    columns = left.columns.union(right.columns)
    for i in ['_left_indicator', '_right_indicator']:
        if i in columns:
            raise ValueError(f'Cannot use `indicator=True` option when data contains a column named {i}')
    if self._indicator_name in columns:
        raise ValueError('Cannot use name of an existing column for indicator column')
    left = left.copy()
    right = right.copy()
    left['_left_indicator'] = 1
    left['_left_indicator'] = left['_left_indicator'].astype('int8')
    right['_right_indicator'] = 2
    right['_right_indicator'] = right['_right_indicator'].astype('int8')
    return (left, right)