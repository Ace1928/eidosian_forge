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
@staticmethod
def _query_iterator(cursor, chunksize: int, columns, index_col=None, coerce_float: bool=True, parse_dates=None, dtype: DtypeArg | None=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy'):
    """Return generator through chunked result set"""
    has_read_data = False
    while True:
        data = cursor.fetchmany(chunksize)
        if type(data) == tuple:
            data = list(data)
        if not data:
            cursor.close()
            if not has_read_data:
                result = DataFrame.from_records([], columns=columns, coerce_float=coerce_float)
                if dtype:
                    result = result.astype(dtype)
                yield result
            break
        has_read_data = True
        yield _wrap_result(data, columns, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, dtype=dtype, dtype_backend=dtype_backend)