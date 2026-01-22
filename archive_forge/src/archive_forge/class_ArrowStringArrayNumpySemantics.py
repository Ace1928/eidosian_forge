from __future__ import annotations
from functools import partial
import operator
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas.compat import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.numeric import NumericDtype
from pandas.core.arrays.string_ import (
from pandas.core.ops import invalid_comparison
from pandas.core.strings.object_array import ObjectStringArrayMixin
class ArrowStringArrayNumpySemantics(ArrowStringArray):
    _storage = 'pyarrow_numpy'

    @classmethod
    def _result_converter(cls, values, na=None):
        if not isna(na):
            values = values.fill_null(bool(na))
        return ArrowExtensionArray(values).to_numpy(na_value=np.nan)

    def __getattribute__(self, item):
        if item in ArrowStringArrayMixin.__dict__ and item not in ('_pa_array', '__dict__'):
            return partial(getattr(ArrowStringArrayMixin, item), self)
        return super().__getattribute__(item)

    def _str_map(self, f, na_value=None, dtype: Dtype | None=None, convert: bool=True):
        if dtype is None:
            dtype = self.dtype
        if na_value is None:
            na_value = self.dtype.na_value
        mask = isna(self)
        arr = np.asarray(self)
        if is_integer_dtype(dtype) or is_bool_dtype(dtype):
            if is_integer_dtype(dtype):
                na_value = np.nan
            else:
                na_value = False
            try:
                result = lib.map_infer_mask(arr, f, mask.view('uint8'), convert=False, na_value=na_value, dtype=np.dtype(dtype))
                return result
            except ValueError:
                result = lib.map_infer_mask(arr, f, mask.view('uint8'), convert=False, na_value=na_value)
                if convert and result.dtype == object:
                    result = lib.maybe_convert_objects(result)
                return result
        elif is_string_dtype(dtype) and (not is_object_dtype(dtype)):
            result = lib.map_infer_mask(arr, f, mask.view('uint8'), convert=False, na_value=na_value)
            result = pa.array(result, mask=mask, type=pa.string(), from_pandas=True)
            return type(self)(result)
        else:
            return lib.map_infer_mask(arr, f, mask.view('uint8'))

    def _convert_int_dtype(self, result):
        if isinstance(result, pa.Array):
            result = result.to_numpy(zero_copy_only=False)
        else:
            result = result.to_numpy()
        if result.dtype == np.int32:
            result = result.astype(np.int64)
        return result

    def _cmp_method(self, other, op):
        try:
            result = super()._cmp_method(other, op)
        except pa.ArrowNotImplementedError:
            return invalid_comparison(self, other, op)
        if op == operator.ne:
            return result.to_numpy(np.bool_, na_value=True)
        else:
            return result.to_numpy(np.bool_, na_value=False)

    def value_counts(self, dropna: bool=True) -> Series:
        from pandas import Series
        result = super().value_counts(dropna)
        return Series(result._values.to_numpy(), index=result.index, name=result.name, copy=False)

    def _reduce(self, name: str, *, skipna: bool=True, keepdims: bool=False, **kwargs):
        if name in ['any', 'all']:
            if not skipna and name == 'all':
                nas = pc.invert(pc.is_null(self._pa_array))
                arr = pc.and_kleene(nas, pc.not_equal(self._pa_array, ''))
            else:
                arr = pc.not_equal(self._pa_array, '')
            return ArrowExtensionArray(arr)._reduce(name, skipna=skipna, keepdims=keepdims, **kwargs)
        else:
            return super()._reduce(name, skipna=skipna, keepdims=keepdims, **kwargs)

    def insert(self, loc: int, item) -> ArrowStringArrayNumpySemantics:
        if item is np.nan:
            item = libmissing.NA
        return super().insert(loc, item)