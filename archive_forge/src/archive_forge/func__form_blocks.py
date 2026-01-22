from __future__ import annotations
from collections.abc import (
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.ops import (
def _form_blocks(arrays: list[ArrayLike], consolidate: bool, refs: list) -> list[Block]:
    tuples = list(enumerate(arrays))
    if not consolidate:
        return _tuples_to_blocks_no_consolidate(tuples, refs)
    grouper = itertools.groupby(tuples, _grouping_func)
    nbs: list[Block] = []
    for (_, dtype), tup_block in grouper:
        block_type = get_block_type(dtype)
        if isinstance(dtype, np.dtype):
            is_dtlike = dtype.kind in 'mM'
            if issubclass(dtype.type, (str, bytes)):
                dtype = np.dtype(object)
            values, placement = _stack_arrays(list(tup_block), dtype)
            if is_dtlike:
                values = ensure_wrapped_if_datetimelike(values)
            blk = block_type(values, placement=BlockPlacement(placement), ndim=2)
            nbs.append(blk)
        elif is_1d_only_ea_dtype(dtype):
            dtype_blocks = [block_type(x[1], placement=BlockPlacement(x[0]), ndim=2) for x in tup_block]
            nbs.extend(dtype_blocks)
        else:
            dtype_blocks = [block_type(ensure_block_shape(x[1], 2), placement=BlockPlacement(x[0]), ndim=2) for x in tup_block]
            nbs.extend(dtype_blocks)
    return nbs