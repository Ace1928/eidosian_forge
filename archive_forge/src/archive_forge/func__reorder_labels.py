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
def _reorder_labels(self, row_positions=None, col_positions=None):
    """
        Reorder the column and or rows in this DataFrame.

        Parameters
        ----------
        row_positions : list of int, optional
            The ordered list of new row orders such that each position within the list
            indicates the new position.
        col_positions : list of int, optional
            The ordered list of new column orders such that each position within the
            list indicates the new position.

        Returns
        -------
        PandasDataframe
            A new PandasDataframe with reordered columns and/or rows.
        """
    new_dtypes = self.copy_dtypes_cache()
    if row_positions is not None:
        ordered_rows = self._partition_mgr_cls.map_axis_partitions(0, self._partitions, lambda df: df.iloc[row_positions], keep_partitioning=True)
        row_idx = self.index[row_positions]
        if len(row_idx) != len(self.index):
            new_lengths = get_length_list(axis_len=len(row_idx), num_splits=ordered_rows.shape[0], min_block_size=MinPartitionSize.get())
        else:
            new_lengths = self._row_lengths_cache
    else:
        ordered_rows = self._partitions
        row_idx = self.copy_index_cache(copy_lengths=True)
        new_lengths = self._row_lengths_cache
    if col_positions is not None:
        ordered_cols = self._partition_mgr_cls.map_axis_partitions(1, ordered_rows, lambda df: df.iloc[:, col_positions], keep_partitioning=True)
        col_idx = self.columns[col_positions]
        if self.has_materialized_dtypes:
            new_dtypes = self.dtypes.iloc[col_positions]
        elif isinstance(self._dtypes, ModinDtypes):
            try:
                new_dtypes = self._dtypes.lazy_get(col_idx)
            except NotImplementedError:
                new_dtypes = None
        if len(col_idx) != len(self.columns):
            new_widths = get_length_list(axis_len=len(col_idx), num_splits=ordered_cols.shape[1], min_block_size=MinPartitionSize.get())
        else:
            new_widths = self._column_widths_cache
    else:
        ordered_cols = ordered_rows
        col_idx = self.copy_columns_cache(copy_lengths=True)
        new_widths = self._column_widths_cache
    return self.__constructor__(ordered_cols, row_idx, col_idx, new_lengths, new_widths, new_dtypes)