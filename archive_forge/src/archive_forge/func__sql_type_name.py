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
def _sql_type_name(self, col):
    dtype: DtypeArg = self.dtype or {}
    if is_dict_like(dtype):
        dtype = cast(dict, dtype)
        if col.name in dtype:
            return dtype[col.name]
    col_type = lib.infer_dtype(col, skipna=True)
    if col_type == 'timedelta64':
        warnings.warn("the 'timedelta' type is not supported, and will be written as integer values (ns frequency) to the database.", UserWarning, stacklevel=find_stack_level())
        col_type = 'integer'
    elif col_type == 'datetime64':
        col_type = 'datetime'
    elif col_type == 'empty':
        col_type = 'string'
    elif col_type == 'complex':
        raise ValueError('Complex datatypes not supported')
    if col_type not in _SQL_TYPES:
        col_type = 'string'
    return _SQL_TYPES[col_type]