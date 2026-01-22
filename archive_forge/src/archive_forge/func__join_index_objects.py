import datetime
import re
from typing import TYPE_CHECKING, Callable, Dict, Hashable, List, Optional, Union
import numpy as np
import pandas
from pandas._libs.lib import no_default
from pandas.api.types import is_object_dtype
from pandas.core.dtypes.common import is_dtype_equal, is_list_like, is_numeric_dtype
from pandas.core.indexes.api import Index, RangeIndex
from modin.config import Engine, IsRayCluster, MinPartitionSize, NPartitions
from modin.core.dataframe.base.dataframe.dataframe import ModinDataframe
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType, is_trivial_index
from modin.core.dataframe.pandas.dataframe.utils import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats.pandas.parsers import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.storage_formats.pandas.utils import get_length_list
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.indexing import is_range_like
from modin.pandas.utils import check_both_not_none, is_full_grab_slice
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
@staticmethod
def _join_index_objects(axis, indexes, how, sort, fill_value=None):
    """
        Join the pair of index objects (columns or rows) by a given strategy.

        Unlike Index.join() in pandas, if `axis` is 1, `sort` is False,
        and `how` is "outer", the result will _not_ be sorted.

        Parameters
        ----------
        axis : {0, 1}
            The axis index object to join (0 - rows, 1 - columns).
        indexes : list(Index)
            The indexes to join on.
        how : {'left', 'right', 'inner', 'outer', None}
            The type of join to join to make. If `None` then joined index
            considered to be the first index in the `indexes` list.
        sort : boolean
            Whether or not to sort the joined index.
        fill_value : any, default: None
            Value to use for missing values.

        Returns
        -------
        (Index, func)
            Joined index with make_reindexer func.
        """
    assert isinstance(indexes, list)

    def merge(left_index, right_index):
        """Combine a pair of indices depending on `axis`, `how` and `sort` from outside."""
        if axis == 1 and how == 'outer' and (not sort):
            return left_index.union(right_index, sort=False)
        else:
            return left_index.join(right_index, how=how, sort=sort)
    all_indices_equal = all((indexes[0].equals(index) for index in indexes[1:]))
    do_join_index = how is not None and (not all_indices_equal)
    need_indexers = axis == 0 and (not all_indices_equal) and any((not index.is_unique for index in indexes))
    indexers = None
    if do_join_index:
        if len(indexes) == 2 and need_indexers:
            indexers = [None, None]
            joined_index, indexers[0], indexers[1] = indexes[0].join(indexes[1], how=how, sort=sort, return_indexers=True)
        else:
            joined_index = indexes[0]
            for index in indexes[1:]:
                joined_index = merge(joined_index, index)
    else:
        joined_index = indexes[0].copy()
    if need_indexers and indexers is None:
        indexers = [index.get_indexer_for(joined_index) for index in indexes]

    def make_reindexer(do_reindex: bool, frame_idx: int):
        """Create callback that reindexes the dataframe using newly computed index."""
        if not do_reindex:
            return lambda df: df
        if need_indexers:
            assert indexers is not None
            return lambda df: df._reindex_with_indexers({0: [joined_index, indexers[frame_idx]]}, copy=True, allow_dups=True, fill_value=fill_value)
        return lambda df: df.reindex(joined_index, axis=axis, fill_value=fill_value)
    return (joined_index, make_reindexer)