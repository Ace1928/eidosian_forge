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
def _get_empty_dtype(join_units: Sequence[JoinUnit]) -> tuple[DtypeObj, DtypeObj]:
    """
    Return dtype and N/A values to use when concatenating specified units.

    Returned N/A value may be None which means there was no casting involved.

    Returns
    -------
    dtype
    """
    if lib.dtypes_all_equal([ju.block.dtype for ju in join_units]):
        empty_dtype = join_units[0].block.dtype
        return (empty_dtype, empty_dtype)
    has_none_blocks = any((unit.block.dtype.kind == 'V' for unit in join_units))
    dtypes = [unit.block.dtype for unit in join_units if not unit.is_na]
    if not len(dtypes):
        dtypes = [unit.block.dtype for unit in join_units if unit.block.dtype.kind != 'V']
    dtype = find_common_type(dtypes)
    if has_none_blocks:
        dtype = ensure_dtype_can_hold_na(dtype)
    dtype_future = dtype
    if len(dtypes) != len(join_units):
        dtypes_future = [unit.block.dtype for unit in join_units if not unit.is_na_after_size_and_isna_all_deprecation]
        if not len(dtypes_future):
            dtypes_future = [unit.block.dtype for unit in join_units if unit.block.dtype.kind != 'V']
        if len(dtypes) != len(dtypes_future):
            dtype_future = find_common_type(dtypes_future)
            if has_none_blocks:
                dtype_future = ensure_dtype_can_hold_na(dtype_future)
    return (dtype, dtype_future)