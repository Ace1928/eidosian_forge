from __future__ import annotations
from collections import defaultdict
from copy import copy
import csv
import datetime
from enum import Enum
import itertools
from typing import (
import warnings
import numpy as np
from pandas._libs import (
import pandas._libs.ops as libops
from pandas._libs.parsers import STR_NA_VALUES
from pandas._libs.tslibs import parsing
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import (
from pandas.core import algorithms
from pandas.core.arrays import (
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools
from pandas.io.common import is_potential_multi_index
@final
def _infer_types(self, values, na_values, no_dtype_specified, try_num_bool: bool=True) -> tuple[ArrayLike, int]:
    """
        Infer types of values, possibly casting

        Parameters
        ----------
        values : ndarray
        na_values : set
        no_dtype_specified: Specifies if we want to cast explicitly
        try_num_bool : bool, default try
           try to cast values to numeric (first preference) or boolean

        Returns
        -------
        converted : ndarray or ExtensionArray
        na_count : int
        """
    na_count = 0
    if issubclass(values.dtype.type, (np.number, np.bool_)):
        na_values = np.array([val for val in na_values if not isinstance(val, str)])
        mask = algorithms.isin(values, na_values)
        na_count = mask.astype('uint8', copy=False).sum()
        if na_count > 0:
            if is_integer_dtype(values):
                values = values.astype(np.float64)
            np.putmask(values, mask, np.nan)
        return (values, na_count)
    dtype_backend = self.dtype_backend
    non_default_dtype_backend = no_dtype_specified and dtype_backend is not lib.no_default
    result: ArrayLike
    if try_num_bool and is_object_dtype(values.dtype):
        try:
            result, result_mask = lib.maybe_convert_numeric(values, na_values, False, convert_to_masked_nullable=non_default_dtype_backend)
        except (ValueError, TypeError):
            na_count = parsers.sanitize_objects(values, na_values)
            result = values
        else:
            if non_default_dtype_backend:
                if result_mask is None:
                    result_mask = np.zeros(result.shape, dtype=np.bool_)
                if result_mask.all():
                    result = IntegerArray(np.ones(result_mask.shape, dtype=np.int64), result_mask)
                elif is_integer_dtype(result):
                    result = IntegerArray(result, result_mask)
                elif is_bool_dtype(result):
                    result = BooleanArray(result, result_mask)
                elif is_float_dtype(result):
                    result = FloatingArray(result, result_mask)
                na_count = result_mask.sum()
            else:
                na_count = isna(result).sum()
    else:
        result = values
        if values.dtype == np.object_:
            na_count = parsers.sanitize_objects(values, na_values)
    if result.dtype == np.object_ and try_num_bool:
        result, bool_mask = libops.maybe_convert_bool(np.asarray(values), true_values=self.true_values, false_values=self.false_values, convert_to_masked_nullable=non_default_dtype_backend)
        if result.dtype == np.bool_ and non_default_dtype_backend:
            if bool_mask is None:
                bool_mask = np.zeros(result.shape, dtype=np.bool_)
            result = BooleanArray(result, bool_mask)
        elif result.dtype == np.object_ and non_default_dtype_backend:
            if not lib.is_datetime_array(result, skipna=True):
                dtype = StringDtype()
                cls = dtype.construct_array_type()
                result = cls._from_sequence(values, dtype=dtype)
    if dtype_backend == 'pyarrow':
        pa = import_optional_dependency('pyarrow')
        if isinstance(result, np.ndarray):
            result = ArrowExtensionArray(pa.array(result, from_pandas=True))
        elif isinstance(result, BaseMaskedArray):
            if result._mask.all():
                result = ArrowExtensionArray(pa.array([None] * len(result)))
            else:
                result = ArrowExtensionArray(pa.array(result._data, mask=result._mask))
        else:
            result = ArrowExtensionArray(pa.array(result.to_numpy(), from_pandas=True))
    return (result, na_count)