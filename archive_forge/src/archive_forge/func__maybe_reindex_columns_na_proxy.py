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
def _maybe_reindex_columns_na_proxy(axes: list[Index], mgrs_indexers: list[tuple[BlockManager, dict[int, np.ndarray]]], needs_copy: bool) -> list[BlockManager]:
    """
    Reindex along columns so that all of the BlockManagers being concatenated
    have matching columns.

    Columns added in this reindexing have dtype=np.void, indicating they
    should be ignored when choosing a column's final dtype.
    """
    new_mgrs = []
    for mgr, indexers in mgrs_indexers:
        for i, indexer in indexers.items():
            mgr = mgr.reindex_indexer(axes[i], indexers[i], axis=i, copy=False, only_slice=True, allow_dups=True, use_na_proxy=True)
        if needs_copy and (not indexers):
            mgr = mgr.copy()
        new_mgrs.append(mgr)
    return new_mgrs