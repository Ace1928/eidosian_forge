from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.missing import NA
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.internals.array_manager import ArrayManager
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def _concatenate_join_units(join_units: list[JoinUnit], copy: bool) -> ArrayLike:
    """
    Concatenate values from several join units along axis=1.
    """
    empty_dtype, empty_dtype_future = _get_empty_dtype(join_units)
    has_none_blocks = any((unit.block.dtype.kind == 'V' for unit in join_units))
    upcasted_na = _dtype_to_na_value(empty_dtype, has_none_blocks)
    to_concat = [ju.get_reindexed_values(empty_dtype=empty_dtype, upcasted_na=upcasted_na) for ju in join_units]
    if any((is_1d_only_ea_dtype(t.dtype) for t in to_concat)):
        to_concat = [t if is_1d_only_ea_dtype(t.dtype) else t[0, :] for t in to_concat]
        concat_values = concat_compat(to_concat, axis=0, ea_compat_axis=True)
        concat_values = ensure_block_shape(concat_values, 2)
    else:
        concat_values = concat_compat(to_concat, axis=1)
    if empty_dtype != empty_dtype_future:
        if empty_dtype == concat_values.dtype:
            warnings.warn('The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.', FutureWarning, stacklevel=find_stack_level())
    return concat_values