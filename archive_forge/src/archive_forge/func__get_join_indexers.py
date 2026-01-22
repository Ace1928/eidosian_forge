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
def _get_join_indexers(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """return the join indexers"""
    left_values = self.left.index._values if self.left_index else self.left_join_keys[-1]
    right_values = self.right.index._values if self.right_index else self.right_join_keys[-1]
    assert left_values.dtype == right_values.dtype
    tolerance = self.tolerance
    if tolerance is not None:
        if needs_i8_conversion(left_values.dtype) or (isinstance(left_values, ArrowExtensionArray) and left_values.dtype.kind in 'mM'):
            tolerance = Timedelta(tolerance)
            if left_values.dtype.kind in 'mM':
                if isinstance(left_values, ArrowExtensionArray):
                    unit = left_values.dtype.pyarrow_dtype.unit
                else:
                    unit = ensure_wrapped_if_datetimelike(left_values).unit
                tolerance = tolerance.as_unit(unit)
            tolerance = tolerance._value
    left_values = self._convert_values_for_libjoin(left_values, 'left')
    right_values = self._convert_values_for_libjoin(right_values, 'right')
    if self.left_by is not None:
        if self.left_index and self.right_index:
            left_join_keys = self.left_join_keys
            right_join_keys = self.right_join_keys
        else:
            left_join_keys = self.left_join_keys[0:-1]
            right_join_keys = self.right_join_keys[0:-1]
        mapped = [_factorize_keys(left_join_keys[n], right_join_keys[n], sort=False) for n in range(len(left_join_keys))]
        if len(left_join_keys) == 1:
            left_by_values = mapped[0][0]
            right_by_values = mapped[0][1]
        else:
            arrs = [np.concatenate(m[:2]) for m in mapped]
            shape = tuple((m[2] for m in mapped))
            group_index = get_group_index(arrs, shape=shape, sort=False, xnull=False)
            left_len = len(left_join_keys[0])
            left_by_values = group_index[:left_len]
            right_by_values = group_index[left_len:]
        left_by_values = ensure_int64(left_by_values)
        right_by_values = ensure_int64(right_by_values)
        func = _asof_by_function(self.direction)
        return func(left_values, right_values, left_by_values, right_by_values, self.allow_exact_matches, tolerance)
    else:
        func = _asof_by_function(self.direction)
        return func(left_values, right_values, None, None, self.allow_exact_matches, tolerance, False)