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
def _maybe_require_matching_dtypes(self, left_join_keys: list[ArrayLike], right_join_keys: list[ArrayLike]) -> None:

    def _check_dtype_match(left: ArrayLike, right: ArrayLike, i: int):
        if left.dtype != right.dtype:
            if isinstance(left.dtype, CategoricalDtype) and isinstance(right.dtype, CategoricalDtype):
                msg = f'incompatible merge keys [{i}] {repr(left.dtype)} and {repr(right.dtype)}, both sides category, but not equal ones'
            else:
                msg = f'incompatible merge keys [{i}] {repr(left.dtype)} and {repr(right.dtype)}, must be the same type'
            raise MergeError(msg)
    for i, (lk, rk) in enumerate(zip(left_join_keys, right_join_keys)):
        _check_dtype_match(lk, rk, i)
    if self.left_index:
        lt = self.left.index._values
    else:
        lt = left_join_keys[-1]
    if self.right_index:
        rt = self.right.index._values
    else:
        rt = right_join_keys[-1]
    _check_dtype_match(lt, rt, 0)