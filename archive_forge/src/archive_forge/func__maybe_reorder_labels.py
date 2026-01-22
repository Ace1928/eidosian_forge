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
def _maybe_reorder_labels(self, intermediate: 'PandasDataframe', row_positions, col_positions) -> 'PandasDataframe':
    """
        Call re-order labels on take_2d_labels_or_positional result if necessary.

        Parameters
        ----------
        intermediate : PandasDataFrame
        row_positions : list-like of ints, optional
            The row positions to extract.
        col_positions : list-like of ints, optional
            The column positions to extract.

        Returns
        -------
        PandasDataframe
        """
    if (row_positions is None or (is_range_like(row_positions) and row_positions.step > 0) or len(row_positions) == 1 or np.all(row_positions[1:] >= row_positions[:-1])) and (col_positions is None or (is_range_like(col_positions) and col_positions.step > 0) or len(col_positions) == 1 or np.all(col_positions[1:] >= col_positions[:-1])):
        return intermediate
    new_row_order, new_col_order = (None, None)
    if is_range_like(row_positions):
        if row_positions.step < 0:
            new_row_order = pandas.RangeIndex(len(row_positions) - 1, -1, -1)
    elif row_positions is not None:
        new_row_order = np.argsort(np.argsort(np.asarray(row_positions, dtype=np.intp)))
    if is_range_like(col_positions):
        if col_positions.step < 0:
            new_col_order = pandas.RangeIndex(len(col_positions) - 1, -1, -1)
    elif col_positions is not None:
        new_col_order = np.argsort(np.argsort(np.asarray(col_positions, dtype=np.intp)))
    return intermediate._reorder_labels(row_positions=new_row_order, col_positions=new_col_order)