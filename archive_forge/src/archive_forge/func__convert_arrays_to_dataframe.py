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
def _convert_arrays_to_dataframe(data, columns, coerce_float: bool=True, dtype_backend: DtypeBackend | Literal['numpy']='numpy') -> DataFrame:
    content = lib.to_object_array_tuples(data)
    arrays = convert_object_array(list(content.T), dtype=None, coerce_float=coerce_float, dtype_backend=dtype_backend)
    if dtype_backend == 'pyarrow':
        pa = import_optional_dependency('pyarrow')
        result_arrays = []
        for arr in arrays:
            pa_array = pa.array(arr, from_pandas=True)
            if arr.dtype == 'string':
                pa_array = pa_array.cast(pa.string())
            result_arrays.append(ArrowExtensionArray(pa_array))
        arrays = result_arrays
    if arrays:
        df = DataFrame(dict(zip(list(range(len(columns))), arrays)))
        df.columns = columns
        return df
    else:
        return DataFrame(columns=columns)