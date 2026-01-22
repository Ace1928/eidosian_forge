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
def _prepare_pandas(self, data: DataFrame) -> None:
    data = data.copy()
    if self._write_index:
        temp = data.reset_index()
        if isinstance(temp, DataFrame):
            data = temp
    data = self._check_column_names(data)
    data = _cast_to_stata_types(data)
    data = self._replace_nans(data)
    self._has_value_labels = np.repeat(False, data.shape[1])
    non_cat_value_labels = self._prepare_non_cat_value_labels(data)
    non_cat_columns = [svl.labname for svl in non_cat_value_labels]
    has_non_cat_val_labels = data.columns.isin(non_cat_columns)
    self._has_value_labels |= has_non_cat_val_labels
    self._value_labels.extend(non_cat_value_labels)
    data = self._prepare_categoricals(data)
    self.nobs, self.nvar = data.shape
    self.data = data
    self.varlist = data.columns.tolist()
    dtypes = data.dtypes
    for col in data:
        if col in self._convert_dates:
            continue
        if lib.is_np_dtype(data[col].dtype, 'M'):
            self._convert_dates[col] = 'tc'
    self._convert_dates = _maybe_convert_to_int_keys(self._convert_dates, self.varlist)
    for key in self._convert_dates:
        new_type = _convert_datetime_to_stata_type(self._convert_dates[key])
        dtypes.iloc[key] = np.dtype(new_type)
    self._encode_strings()
    self._set_formats_and_types(dtypes)
    if self._convert_dates is not None:
        for key in self._convert_dates:
            if isinstance(key, int):
                self.fmtlist[key] = self._convert_dates[key]