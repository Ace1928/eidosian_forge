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
def _compute_dtypes(self, columns=None):
    """
        Compute the data types via TreeReduce pattern for the specified columns.

        Parameters
        ----------
        columns : list-like, default: None
            Columns to compute dtypes for. If not specified compute dtypes
            for all the columns in the dataframe.

        Returns
        -------
        pandas.Series
            A pandas Series containing the data types for this dataframe.
        """

    def dtype_builder(df):
        return df.apply(lambda col: find_common_type(col.values), axis=0)
    if columns is not None:
        numeric_indices = sorted(self.columns.get_indexer_for(columns))
        if any((pos < 0 for pos in numeric_indices)):
            raise KeyError(f'Some of the columns are not in index: subset={columns}; columns={self.columns}')
        obj = self.take_2d_labels_or_positional(col_labels=self.columns[numeric_indices].tolist())
    else:
        obj = self
    if len(obj.columns) > 0:
        dtypes = obj.tree_reduce(0, lambda df: df.dtypes, dtype_builder).to_pandas().iloc[0]
    else:
        dtypes = pandas.Series([])
    dtypes.name = None
    return dtypes