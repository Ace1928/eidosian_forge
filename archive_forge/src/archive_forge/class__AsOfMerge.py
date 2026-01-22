from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from typing import (
import uuid
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas._typing import (
from pandas.errors import MergeError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.frame import _merge_doc
from pandas.core.indexes.api import default_index
from pandas.core.sorting import (
class _AsOfMerge(_OrderedMerge):
    _merge_type = 'asof_merge'

    def __init__(self, left: DataFrame | Series, right: DataFrame | Series, on: IndexLabel | None=None, left_on: IndexLabel | None=None, right_on: IndexLabel | None=None, left_index: bool=False, right_index: bool=False, by=None, left_by=None, right_by=None, suffixes: Suffixes=('_x', '_y'), how: Literal['asof']='asof', tolerance=None, allow_exact_matches: bool=True, direction: str='backward') -> None:
        self.by = by
        self.left_by = left_by
        self.right_by = right_by
        self.tolerance = tolerance
        self.allow_exact_matches = allow_exact_matches
        self.direction = direction
        if self.direction not in ['backward', 'forward', 'nearest']:
            raise MergeError(f'direction invalid: {self.direction}')
        if not is_bool(self.allow_exact_matches):
            msg = f'allow_exact_matches must be boolean, passed {self.allow_exact_matches}'
            raise MergeError(msg)
        _OrderedMerge.__init__(self, left, right, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, how=how, suffixes=suffixes, fill_method=None)

    def _validate_left_right_on(self, left_on, right_on):
        left_on, right_on = super()._validate_left_right_on(left_on, right_on)
        if len(left_on) != 1 and (not self.left_index):
            raise MergeError('can only asof on a key for left')
        if len(right_on) != 1 and (not self.right_index):
            raise MergeError('can only asof on a key for right')
        if self.left_index and isinstance(self.left.index, MultiIndex):
            raise MergeError('left can only have one index')
        if self.right_index and isinstance(self.right.index, MultiIndex):
            raise MergeError('right can only have one index')
        if self.by is not None:
            if self.left_by is not None or self.right_by is not None:
                raise MergeError('Can only pass by OR left_by and right_by')
            self.left_by = self.right_by = self.by
        if self.left_by is None and self.right_by is not None:
            raise MergeError('missing left_by')
        if self.left_by is not None and self.right_by is None:
            raise MergeError('missing right_by')
        if not self.left_index:
            left_on_0 = left_on[0]
            if isinstance(left_on_0, _known):
                lo_dtype = left_on_0.dtype
            else:
                lo_dtype = self.left._get_label_or_level_values(left_on_0).dtype if left_on_0 in self.left.columns else self.left.index.get_level_values(left_on_0)
        else:
            lo_dtype = self.left.index.dtype
        if not self.right_index:
            right_on_0 = right_on[0]
            if isinstance(right_on_0, _known):
                ro_dtype = right_on_0.dtype
            else:
                ro_dtype = self.right._get_label_or_level_values(right_on_0).dtype if right_on_0 in self.right.columns else self.right.index.get_level_values(right_on_0)
        else:
            ro_dtype = self.right.index.dtype
        if is_object_dtype(lo_dtype) or is_object_dtype(ro_dtype) or is_string_dtype(lo_dtype) or is_string_dtype(ro_dtype):
            raise MergeError(f'Incompatible merge dtype, {repr(ro_dtype)} and {repr(lo_dtype)}, both sides must have numeric dtype')
        if self.left_by is not None:
            if not is_list_like(self.left_by):
                self.left_by = [self.left_by]
            if not is_list_like(self.right_by):
                self.right_by = [self.right_by]
            if len(self.left_by) != len(self.right_by):
                raise MergeError('left_by and right_by must be the same length')
            left_on = self.left_by + list(left_on)
            right_on = self.right_by + list(right_on)
        return (left_on, right_on)

    def _maybe_require_matching_dtypes(self, left_join_keys: list[ArrayLike], right_join_keys: list[ArrayLike]) -> None:

        def _check_dtype_match(left: ArrayLike, right: ArrayLike, i: int):
            if left.dtype != right.dtype:
                if isinstance(left.dtype, CategoricalDtype) and isinstance(right.dtype, CategoricalDtype):
                    msg = f'incompatible merge keys [{i}] {repr(left.dtype)} and {repr(right.dtype)}, both sides category, but not equal ones'
                else:
                    msg = f'incompatible merge keys [{i}] {repr(left.dtype)} and {repr(right.dtype)}, must be the same type'
                raise MergeError(msg)
        for i, (lk, rk) in enumerate(zip(left_join_keys, right_join_keys)):
            _check_dtype_match(lk, rk, i)
        if self.left_index:
            lt = self.left.index._values
        else:
            lt = left_join_keys[-1]
        if self.right_index:
            rt = self.right.index._values
        else:
            rt = right_join_keys[-1]
        _check_dtype_match(lt, rt, 0)

    def _validate_tolerance(self, left_join_keys: list[ArrayLike]) -> None:
        if self.tolerance is not None:
            if self.left_index:
                lt = self.left.index._values
            else:
                lt = left_join_keys[-1]
            msg = f'incompatible tolerance {self.tolerance}, must be compat with type {repr(lt.dtype)}'
            if needs_i8_conversion(lt.dtype) or (isinstance(lt, ArrowExtensionArray) and lt.dtype.kind in 'mM'):
                if not isinstance(self.tolerance, datetime.timedelta):
                    raise MergeError(msg)
                if self.tolerance < Timedelta(0):
                    raise MergeError('tolerance must be positive')
            elif is_integer_dtype(lt.dtype):
                if not is_integer(self.tolerance):
                    raise MergeError(msg)
                if self.tolerance < 0:
                    raise MergeError('tolerance must be positive')
            elif is_float_dtype(lt.dtype):
                if not is_number(self.tolerance):
                    raise MergeError(msg)
                if self.tolerance < 0:
                    raise MergeError('tolerance must be positive')
            else:
                raise MergeError('key must be integer, timestamp or float')

    def _convert_values_for_libjoin(self, values: AnyArrayLike, side: str) -> np.ndarray:
        if not Index(values).is_monotonic_increasing:
            if isna(values).any():
                raise ValueError(f'Merge keys contain null values on {side} side')
            raise ValueError(f'{side} keys must be sorted')
        if isinstance(values, ArrowExtensionArray):
            values = values._maybe_convert_datelike_array()
        if needs_i8_conversion(values.dtype):
            values = values.view('i8')
        elif isinstance(values, BaseMaskedArray):
            values = values._data
        elif isinstance(values, ExtensionArray):
            values = values.to_numpy()
        return values

    def _get_join_indexers(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """return the join indexers"""
        left_values = self.left.index._values if self.left_index else self.left_join_keys[-1]
        right_values = self.right.index._values if self.right_index else self.right_join_keys[-1]
        assert left_values.dtype == right_values.dtype
        tolerance = self.tolerance
        if tolerance is not None:
            if needs_i8_conversion(left_values.dtype) or (isinstance(left_values, ArrowExtensionArray) and left_values.dtype.kind in 'mM'):
                tolerance = Timedelta(tolerance)
                if left_values.dtype.kind in 'mM':
                    if isinstance(left_values, ArrowExtensionArray):
                        unit = left_values.dtype.pyarrow_dtype.unit
                    else:
                        unit = ensure_wrapped_if_datetimelike(left_values).unit
                    tolerance = tolerance.as_unit(unit)
                tolerance = tolerance._value
        left_values = self._convert_values_for_libjoin(left_values, 'left')
        right_values = self._convert_values_for_libjoin(right_values, 'right')
        if self.left_by is not None:
            if self.left_index and self.right_index:
                left_join_keys = self.left_join_keys
                right_join_keys = self.right_join_keys
            else:
                left_join_keys = self.left_join_keys[0:-1]
                right_join_keys = self.right_join_keys[0:-1]
            mapped = [_factorize_keys(left_join_keys[n], right_join_keys[n], sort=False) for n in range(len(left_join_keys))]
            if len(left_join_keys) == 1:
                left_by_values = mapped[0][0]
                right_by_values = mapped[0][1]
            else:
                arrs = [np.concatenate(m[:2]) for m in mapped]
                shape = tuple((m[2] for m in mapped))
                group_index = get_group_index(arrs, shape=shape, sort=False, xnull=False)
                left_len = len(left_join_keys[0])
                left_by_values = group_index[:left_len]
                right_by_values = group_index[left_len:]
            left_by_values = ensure_int64(left_by_values)
            right_by_values = ensure_int64(right_by_values)
            func = _asof_by_function(self.direction)
            return func(left_values, right_values, left_by_values, right_by_values, self.allow_exact_matches, tolerance)
        else:
            func = _asof_by_function(self.direction)
            return func(left_values, right_values, None, None, self.allow_exact_matches, tolerance, False)