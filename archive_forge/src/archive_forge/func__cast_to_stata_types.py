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
def _cast_to_stata_types(data: DataFrame) -> DataFrame:
    """
    Checks the dtypes of the columns of a pandas DataFrame for
    compatibility with the data types and ranges supported by Stata, and
    converts if necessary.

    Parameters
    ----------
    data : DataFrame
        The DataFrame to check and convert

    Notes
    -----
    Numeric columns in Stata must be one of int8, int16, int32, float32 or
    float64, with some additional value restrictions.  int8 and int16 columns
    are checked for violations of the value restrictions and upcast if needed.
    int64 data is not usable in Stata, and so it is downcast to int32 whenever
    the value are in the int32 range, and sidecast to float64 when larger than
    this range.  If the int64 values are outside of the range of those
    perfectly representable as float64 values, a warning is raised.

    bool columns are cast to int8.  uint columns are converted to int of the
    same size if there is no loss in precision, otherwise are upcast to a
    larger type.  uint64 is currently not supported since it is concerted to
    object in a DataFrame.
    """
    ws = ''
    conversion_data: tuple[tuple[type, type, type], tuple[type, type, type], tuple[type, type, type], tuple[type, type, type], tuple[type, type, type]] = ((np.bool_, np.int8, np.int8), (np.uint8, np.int8, np.int16), (np.uint16, np.int16, np.int32), (np.uint32, np.int32, np.int64), (np.uint64, np.int64, np.float64))
    float32_max = struct.unpack('<f', b'\xff\xff\xff~')[0]
    float64_max = struct.unpack('<d', b'\xff\xff\xff\xff\xff\xff\xdf\x7f')[0]
    for col in data:
        is_nullable_int = isinstance(data[col].dtype, ExtensionDtype) and data[col].dtype.kind in 'iub'
        orig_missing = data[col].isna()
        if is_nullable_int:
            fv = 0 if data[col].dtype.kind in 'iu' else False
            data[col] = data[col].fillna(fv).astype(data[col].dtype.numpy_dtype)
        elif isinstance(data[col].dtype, ExtensionDtype):
            if getattr(data[col].dtype, 'numpy_dtype', None) is not None:
                data[col] = data[col].astype(data[col].dtype.numpy_dtype)
            elif is_string_dtype(data[col].dtype):
                data[col] = data[col].astype('object')
        dtype = data[col].dtype
        empty_df = data.shape[0] == 0
        for c_data in conversion_data:
            if dtype == c_data[0]:
                if empty_df or data[col].max() <= np.iinfo(c_data[1]).max:
                    dtype = c_data[1]
                else:
                    dtype = c_data[2]
                if c_data[2] == np.int64:
                    if data[col].max() >= 2 ** 53:
                        ws = precision_loss_doc.format('uint64', 'float64')
                data[col] = data[col].astype(dtype)
        if dtype == np.int8 and (not empty_df):
            if data[col].max() > 100 or data[col].min() < -127:
                data[col] = data[col].astype(np.int16)
        elif dtype == np.int16 and (not empty_df):
            if data[col].max() > 32740 or data[col].min() < -32767:
                data[col] = data[col].astype(np.int32)
        elif dtype == np.int64:
            if empty_df or (data[col].max() <= 2147483620 and data[col].min() >= -2147483647):
                data[col] = data[col].astype(np.int32)
            else:
                data[col] = data[col].astype(np.float64)
                if data[col].max() >= 2 ** 53 or data[col].min() <= -2 ** 53:
                    ws = precision_loss_doc.format('int64', 'float64')
        elif dtype in (np.float32, np.float64):
            if np.isinf(data[col]).any():
                raise ValueError(f'Column {col} contains infinity or -infinitywhich is outside the range supported by Stata.')
            value = data[col].max()
            if dtype == np.float32 and value > float32_max:
                data[col] = data[col].astype(np.float64)
            elif dtype == np.float64:
                if value > float64_max:
                    raise ValueError(f'Column {col} has a maximum value ({value}) outside the range supported by Stata ({float64_max})')
        if is_nullable_int:
            if orig_missing.any():
                sentinel = StataMissingValue.BASE_MISSING_VALUES[data[col].dtype.name]
                data.loc[orig_missing, col] = sentinel
    if ws:
        warnings.warn(ws, PossiblePrecisionLoss, stacklevel=find_stack_level())
    return data