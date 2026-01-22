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
def _get_join_info(self) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
    left_ax = self.left.index
    right_ax = self.right.index
    if self.left_index and self.right_index and (self.how != 'asof'):
        join_index, left_indexer, right_indexer = left_ax.join(right_ax, how=self.how, return_indexers=True, sort=self.sort)
    elif self.right_index and self.how == 'left':
        join_index, left_indexer, right_indexer = _left_join_on_index(left_ax, right_ax, self.left_join_keys, sort=self.sort)
    elif self.left_index and self.how == 'right':
        join_index, right_indexer, left_indexer = _left_join_on_index(right_ax, left_ax, self.right_join_keys, sort=self.sort)
    else:
        left_indexer, right_indexer = self._get_join_indexers()
        if self.right_index:
            if len(self.left) > 0:
                join_index = self._create_join_index(left_ax, right_ax, left_indexer, how='right')
            elif right_indexer is None:
                join_index = right_ax.copy()
            else:
                join_index = right_ax.take(right_indexer)
        elif self.left_index:
            if self.how == 'asof':
                join_index = self._create_join_index(left_ax, right_ax, left_indexer, how='left')
            elif len(self.right) > 0:
                join_index = self._create_join_index(right_ax, left_ax, right_indexer, how='left')
            elif left_indexer is None:
                join_index = left_ax.copy()
            else:
                join_index = left_ax.take(left_indexer)
        else:
            n = len(left_ax) if left_indexer is None else len(left_indexer)
            join_index = default_index(n)
    return (join_index, left_indexer, right_indexer)