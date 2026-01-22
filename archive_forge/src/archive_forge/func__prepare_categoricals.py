from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
def _prepare_categoricals(self, data: DataFrame) -> DataFrame:
    """
        Check for categorical columns, retain categorical information for
        Stata file and convert categorical data to int
        """
    is_cat = [isinstance(dtype, CategoricalDtype) for dtype in data.dtypes]
    if not any(is_cat):
        return data
    self._has_value_labels |= np.array(is_cat)
    get_base_missing_value = StataMissingValue.get_base_missing_value
    data_formatted = []
    for col, col_is_cat in zip(data, is_cat):
        if col_is_cat:
            svl = StataValueLabel(data[col], encoding=self._encoding)
            self._value_labels.append(svl)
            dtype = data[col].cat.codes.dtype
            if dtype == np.int64:
                raise ValueError('It is not possible to export int64-based categorical data to Stata.')
            values = data[col].cat.codes._values.copy()
            if values.max() >= get_base_missing_value(dtype):
                if dtype == np.int8:
                    dtype = np.dtype(np.int16)
                elif dtype == np.int16:
                    dtype = np.dtype(np.int32)
                else:
                    dtype = np.dtype(np.float64)
                values = np.array(values, dtype=dtype)
            values[values == -1] = get_base_missing_value(dtype)
            data_formatted.append((col, values))
        else:
            data_formatted.append((col, data[col]))
    return DataFrame.from_dict(dict(data_formatted))