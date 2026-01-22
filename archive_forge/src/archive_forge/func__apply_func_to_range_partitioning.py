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
@lazy_metadata_decorator(apply_axis='both')
def _apply_func_to_range_partitioning(self, key_columns, func, ascending=True, preserve_columns=False, data=None, data_key_columns=None, level=None, shuffle_func_cls=ShuffleSortFunctions, **kwargs):
    """
        Reshuffle data so it would be range partitioned and then apply the passed function row-wise.

        Parameters
        ----------
        key_columns : list of hashables
            Columns to build the range partitioning for. Can't be specified along with `level`.
        func : callable(pandas.DataFrame) -> pandas.DataFrame
            Function to apply against partitions.
        ascending : bool, default: True
            Whether the range should be built in ascending or descending order.
        preserve_columns : bool, default: False
            If the columns cache should be preserved (specify this flag if `func` doesn't change column labels).
        data : PandasDataframe, optional
            Dataframe to range-partition along with the `self` frame. If specified, the `func` will recieve
            a dataframe with an additional MultiIndex level in columns that separates `self` and `data`:
            ``df["grouper"] # self`` and ``df["data"] # data``.
        data_key_columns : list of hashables, optional
            Additional key columns from `data`. Will be combined with `key_columns`.
        level : list of ints or labels, optional
            Index level(s) to build the range partitioning for. Can't be specified along with `key_columns`.
        shuffle_func_cls : cls, default: ShuffleSortFunctions
            A class implementing ``modin.core.dataframe.pandas.utils.ShuffleFunctions`` to be used
            as a shuffle function.
        **kwargs : dict
            Additional arguments to forward to the range builder function.

        Returns
        -------
        PandasDataframe
            A new dataframe.
        """
    if data is not None:
        new_grouper_cols = pandas.MultiIndex.from_tuples([('grouper', *col) if isinstance(col, tuple) else ('grouper', col) for col in self.columns])
        grouper = self.copy()
        grouper.columns = new_grouper_cols
        new_data_cols = pandas.MultiIndex.from_tuples([('data', *col) if isinstance(col, tuple) else ('data', col) for col in data.columns])
        data = data.copy()
        data.columns = new_data_cols
        grouper = grouper.concat(axis=1, others=[data], how='right', sort=False)
        key_columns = [('grouper', *col) if isinstance(col, tuple) else ('grouper', col) for col in key_columns]
        if data_key_columns is None:
            data_key_columns = []
        else:
            data_key_columns = [('data', *col) if isinstance(col, tuple) else ('data', col) for col in data_key_columns]
        key_columns += data_key_columns
    else:
        grouper = self
    if grouper._partitions.shape[0] == 1:
        result = grouper.apply_full_axis(axis=1, func=func, new_columns=grouper.copy_columns_cache() if preserve_columns else None)
        if preserve_columns:
            result._set_axis_lengths_cache(grouper._column_widths_cache, axis=1)
        return result
    ideal_num_new_partitions = min(len(grouper._partitions), NPartitions.get())
    m = len(grouper) / ideal_num_new_partitions
    sampling_probability = 1 / m * np.log(ideal_num_new_partitions * len(grouper))
    if sampling_probability >= 1:
        from modin.config import MinPartitionSize
        ideal_num_new_partitions = round(len(grouper) / MinPartitionSize.get())
        if len(grouper) < MinPartitionSize.get() or ideal_num_new_partitions < 2:
            return grouper.combine_and_apply(func=func)
        if ideal_num_new_partitions < len(grouper._partitions):
            if len(grouper._partitions) % ideal_num_new_partitions == 0:
                joining_partitions = np.split(grouper._partitions, ideal_num_new_partitions)
            else:
                step = round(len(grouper._partitions) / ideal_num_new_partitions)
                joining_partitions = np.split(grouper._partitions, range(step, len(grouper._partitions), step))
            new_partitions = np.array([grouper._partition_mgr_cls.column_partitions(ptn_grp, full_axis=False) for ptn_grp in joining_partitions])
        else:
            new_partitions = grouper._partitions
    else:
        new_partitions = grouper._partitions
    shuffling_functions = shuffle_func_cls(grouper, key_columns, ascending[0] if is_list_like(ascending) else ascending, ideal_num_new_partitions, level=level, **kwargs)
    if key_columns:
        key_indices = grouper.columns.get_indexer_for(key_columns)
        partition_indices = np.unique(np.digitize(key_indices, np.cumsum(grouper.column_widths)))
    elif level is not None:
        partition_indices = [0]
    else:
        raise ValueError("Must specify either 'level' or 'key_columns'")
    new_partitions = grouper._partition_mgr_cls.shuffle_partitions(new_partitions, partition_indices, shuffling_functions, func)
    result = grouper.__constructor__(new_partitions)
    if preserve_columns:
        result.set_columns_cache(grouper.copy_columns_cache())
        if grouper.has_materialized_columns:
            result._set_axis_lengths_cache([len(grouper.columns)], axis=1)
    return result