from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
def _harmonize_columns(self, parse_dates=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy') -> None:
    """
        Make the DataFrame's column types align with the SQL table
        column types.
        Need to work around limited NA value support. Floats are always
        fine, ints must always be floats if there are Null values.
        Booleans are hard because converting bool column with None replaces
        all Nones with false. Therefore only convert bool if there are no
        NA values.
        Datetimes should already be converted to np.datetime64 if supported,
        but here we also force conversion if required.
        """
    parse_dates = _process_parse_dates_argument(parse_dates)
    for sql_col in self.table.columns:
        col_name = sql_col.name
        try:
            df_col = self.frame[col_name]
            if col_name in parse_dates:
                try:
                    fmt = parse_dates[col_name]
                except TypeError:
                    fmt = None
                self.frame[col_name] = _handle_date_column(df_col, format=fmt)
                continue
            col_type = self._get_dtype(sql_col.type)
            if col_type is datetime or col_type is date or col_type is DatetimeTZDtype:
                utc = col_type is DatetimeTZDtype
                self.frame[col_name] = _handle_date_column(df_col, utc=utc)
            elif dtype_backend == 'numpy' and col_type is float:
                self.frame[col_name] = df_col.astype(col_type, copy=False)
            elif dtype_backend == 'numpy' and len(df_col) == df_col.count():
                if col_type is np.dtype('int64') or col_type is bool:
                    self.frame[col_name] = df_col.astype(col_type, copy=False)
        except KeyError:
            pass