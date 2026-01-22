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
@final
def _maybe_coerce_merge_keys(self) -> None:
    for lk, rk, name in zip(self.left_join_keys, self.right_join_keys, self.join_names):
        if len(lk) and (not len(rk)) or (not len(lk) and len(rk)):
            continue
        lk = extract_array(lk, extract_numpy=True)
        rk = extract_array(rk, extract_numpy=True)
        lk_is_cat = isinstance(lk.dtype, CategoricalDtype)
        rk_is_cat = isinstance(rk.dtype, CategoricalDtype)
        lk_is_object_or_string = is_object_dtype(lk.dtype) or is_string_dtype(lk.dtype)
        rk_is_object_or_string = is_object_dtype(rk.dtype) or is_string_dtype(rk.dtype)
        if lk_is_cat and rk_is_cat:
            lk = cast(Categorical, lk)
            rk = cast(Categorical, rk)
            if lk._categories_match_up_to_permutation(rk):
                continue
        elif lk_is_cat or rk_is_cat:
            pass
        elif lk.dtype == rk.dtype:
            continue
        msg = f"You are trying to merge on {lk.dtype} and {rk.dtype} columns for key '{name}'. If you wish to proceed you should use pd.concat"
        if is_numeric_dtype(lk.dtype) and is_numeric_dtype(rk.dtype):
            if lk.dtype.kind == rk.dtype.kind:
                continue
            if isinstance(lk.dtype, ExtensionDtype) and (not isinstance(rk.dtype, ExtensionDtype)):
                ct = find_common_type([lk.dtype, rk.dtype])
                if isinstance(ct, ExtensionDtype):
                    com_cls = ct.construct_array_type()
                    rk = com_cls._from_sequence(rk, dtype=ct, copy=False)
                else:
                    rk = rk.astype(ct)
            elif isinstance(rk.dtype, ExtensionDtype):
                ct = find_common_type([lk.dtype, rk.dtype])
                if isinstance(ct, ExtensionDtype):
                    com_cls = ct.construct_array_type()
                    lk = com_cls._from_sequence(lk, dtype=ct, copy=False)
                else:
                    lk = lk.astype(ct)
            if is_integer_dtype(rk.dtype) and is_float_dtype(lk.dtype):
                with np.errstate(invalid='ignore'):
                    casted = lk.astype(rk.dtype)
                mask = ~np.isnan(lk)
                match = lk == casted
                if not match[mask].all():
                    warnings.warn('You are merging on int and float columns where the float values are not equal to their int representation.', UserWarning, stacklevel=find_stack_level())
                continue
            if is_float_dtype(rk.dtype) and is_integer_dtype(lk.dtype):
                with np.errstate(invalid='ignore'):
                    casted = rk.astype(lk.dtype)
                mask = ~np.isnan(rk)
                match = rk == casted
                if not match[mask].all():
                    warnings.warn('You are merging on int and float columns where the float values are not equal to their int representation.', UserWarning, stacklevel=find_stack_level())
                continue
            if lib.infer_dtype(lk, skipna=False) == lib.infer_dtype(rk, skipna=False):
                continue
        elif lk_is_object_or_string and is_bool_dtype(rk.dtype) or (is_bool_dtype(lk.dtype) and rk_is_object_or_string):
            pass
        elif lk_is_object_or_string and is_numeric_dtype(rk.dtype) or (is_numeric_dtype(lk.dtype) and rk_is_object_or_string):
            inferred_left = lib.infer_dtype(lk, skipna=False)
            inferred_right = lib.infer_dtype(rk, skipna=False)
            bool_types = ['integer', 'mixed-integer', 'boolean', 'empty']
            string_types = ['string', 'unicode', 'mixed', 'bytes', 'empty']
            if inferred_left in bool_types and inferred_right in bool_types:
                pass
            elif inferred_left in string_types and inferred_right not in string_types or (inferred_right in string_types and inferred_left not in string_types):
                raise ValueError(msg)
        elif needs_i8_conversion(lk.dtype) and (not needs_i8_conversion(rk.dtype)):
            raise ValueError(msg)
        elif not needs_i8_conversion(lk.dtype) and needs_i8_conversion(rk.dtype):
            raise ValueError(msg)
        elif isinstance(lk.dtype, DatetimeTZDtype) and (not isinstance(rk.dtype, DatetimeTZDtype)):
            raise ValueError(msg)
        elif not isinstance(lk.dtype, DatetimeTZDtype) and isinstance(rk.dtype, DatetimeTZDtype):
            raise ValueError(msg)
        elif isinstance(lk.dtype, DatetimeTZDtype) and isinstance(rk.dtype, DatetimeTZDtype) or (lk.dtype.kind == 'M' and rk.dtype.kind == 'M'):
            continue
        elif lk.dtype.kind == 'M' and rk.dtype.kind == 'm':
            raise ValueError(msg)
        elif lk.dtype.kind == 'm' and rk.dtype.kind == 'M':
            raise ValueError(msg)
        elif is_object_dtype(lk.dtype) and is_object_dtype(rk.dtype):
            continue
        if name in self.left.columns:
            typ = cast(Categorical, lk).categories.dtype if lk_is_cat else object
            self.left = self.left.copy()
            self.left[name] = self.left[name].astype(typ)
        if name in self.right.columns:
            typ = cast(Categorical, rk).categories.dtype if rk_is_cat else object
            self.right = self.right.copy()
            self.right[name] = self.right[name].astype(typ)