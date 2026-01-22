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
def _validate_axes_lengths(self):
    """Validate that labels are split correctly if split is known."""
    if self._row_lengths_cache is not None and self.has_materialized_index and (len(self.index) > 0):
        num_rows = sum(self._row_lengths_cache)
        if num_rows > 0:
            ErrorMessage.catch_bugs_and_request_email(num_rows != len(self.index), f'Row lengths: {num_rows} != {len(self.index)}')
        ErrorMessage.catch_bugs_and_request_email(any((val < 0 for val in self._row_lengths_cache)), f'Row lengths cannot be negative: {self._row_lengths_cache}')
    if self._column_widths_cache is not None and self.has_materialized_columns and (len(self.columns) > 0):
        num_columns = sum(self._column_widths_cache)
        if num_columns > 0:
            ErrorMessage.catch_bugs_and_request_email(num_columns != len(self.columns), f'Column widths: {num_columns} != {len(self.columns)}')
        ErrorMessage.catch_bugs_and_request_email(any((val < 0 for val in self._column_widths_cache)), f'Column widths cannot be negative: {self._column_widths_cache}')