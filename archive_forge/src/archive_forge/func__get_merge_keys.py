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
def _get_merge_keys(self) -> tuple[list[ArrayLike], list[ArrayLike], list[Hashable], list[Hashable], list[Hashable]]:
    """
        Returns
        -------
        left_keys, right_keys, join_names, left_drop, right_drop
        """
    left_keys: list[ArrayLike] = []
    right_keys: list[ArrayLike] = []
    join_names: list[Hashable] = []
    right_drop: list[Hashable] = []
    left_drop: list[Hashable] = []
    left, right = (self.left, self.right)
    is_lkey = lambda x: isinstance(x, _known) and len(x) == len(left)
    is_rkey = lambda x: isinstance(x, _known) and len(x) == len(right)
    if _any(self.left_on) and _any(self.right_on):
        for lk, rk in zip(self.left_on, self.right_on):
            lk = extract_array(lk, extract_numpy=True)
            rk = extract_array(rk, extract_numpy=True)
            if is_lkey(lk):
                lk = cast(ArrayLike, lk)
                left_keys.append(lk)
                if is_rkey(rk):
                    rk = cast(ArrayLike, rk)
                    right_keys.append(rk)
                    join_names.append(None)
                else:
                    rk = cast(Hashable, rk)
                    if rk is not None:
                        right_keys.append(right._get_label_or_level_values(rk))
                        join_names.append(rk)
                    else:
                        right_keys.append(right.index._values)
                        join_names.append(right.index.name)
            else:
                if not is_rkey(rk):
                    rk = cast(Hashable, rk)
                    if rk is not None:
                        right_keys.append(right._get_label_or_level_values(rk))
                    else:
                        right_keys.append(right.index._values)
                    if lk is not None and lk == rk:
                        right_drop.append(rk)
                else:
                    rk = cast(ArrayLike, rk)
                    right_keys.append(rk)
                if lk is not None:
                    lk = cast(Hashable, lk)
                    left_keys.append(left._get_label_or_level_values(lk))
                    join_names.append(lk)
                else:
                    left_keys.append(left.index._values)
                    join_names.append(left.index.name)
    elif _any(self.left_on):
        for k in self.left_on:
            if is_lkey(k):
                k = extract_array(k, extract_numpy=True)
                k = cast(ArrayLike, k)
                left_keys.append(k)
                join_names.append(None)
            else:
                k = cast(Hashable, k)
                left_keys.append(left._get_label_or_level_values(k))
                join_names.append(k)
        if isinstance(self.right.index, MultiIndex):
            right_keys = [lev._values.take(lev_codes) for lev, lev_codes in zip(self.right.index.levels, self.right.index.codes)]
        else:
            right_keys = [self.right.index._values]
    elif _any(self.right_on):
        for k in self.right_on:
            k = extract_array(k, extract_numpy=True)
            if is_rkey(k):
                k = cast(ArrayLike, k)
                right_keys.append(k)
                join_names.append(None)
            else:
                k = cast(Hashable, k)
                right_keys.append(right._get_label_or_level_values(k))
                join_names.append(k)
        if isinstance(self.left.index, MultiIndex):
            left_keys = [lev._values.take(lev_codes) for lev, lev_codes in zip(self.left.index.levels, self.left.index.codes)]
        else:
            left_keys = [self.left.index._values]
    return (left_keys, right_keys, join_names, left_drop, right_drop)