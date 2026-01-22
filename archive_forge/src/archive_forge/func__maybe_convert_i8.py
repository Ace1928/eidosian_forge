from __future__ import annotations
from operator import (
import textwrap
from typing import (
import numpy as np
from pandas._libs import lib
from pandas._libs.interval import (
from pandas._libs.tslibs import (
from pandas.errors import InvalidIndexError
from pandas.util._decorators import (
from pandas.util._exceptions import rewrite_exception
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.algorithms import unique
from pandas.core.arrays.datetimelike import validate_periods
from pandas.core.arrays.interval import (
import pandas.core.common as com
from pandas.core.indexers import is_valid_positional_slice
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.extension import (
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.timedeltas import (
def _maybe_convert_i8(self, key):
    """
        Maybe convert a given key to its equivalent i8 value(s). Used as a
        preprocessing step prior to IntervalTree queries (self._engine), which
        expects numeric data.

        Parameters
        ----------
        key : scalar or list-like
            The key that should maybe be converted to i8.

        Returns
        -------
        scalar or list-like
            The original key if no conversion occurred, int if converted scalar,
            Index with an int64 dtype if converted list-like.
        """
    if is_list_like(key):
        key = ensure_index(key)
        key = maybe_upcast_numeric_to_64bit(key)
    if not self._needs_i8_conversion(key):
        return key
    scalar = is_scalar(key)
    key_dtype = getattr(key, 'dtype', None)
    if isinstance(key_dtype, IntervalDtype) or isinstance(key, Interval):
        left = self._maybe_convert_i8(key.left)
        right = self._maybe_convert_i8(key.right)
        constructor = Interval if scalar else IntervalIndex.from_arrays
        return constructor(left, right, closed=self.closed)
    if scalar:
        key_dtype, key_i8 = infer_dtype_from_scalar(key)
        if isinstance(key, Period):
            key_i8 = key.ordinal
        elif isinstance(key_i8, Timestamp):
            key_i8 = key_i8._value
        elif isinstance(key_i8, (np.datetime64, np.timedelta64)):
            key_i8 = key_i8.view('i8')
    else:
        key_dtype, key_i8 = (key.dtype, Index(key.asi8))
        if key.hasnans:
            key_i8 = key_i8.where(~key._isnan)
    subtype = self.dtype.subtype
    if subtype != key_dtype:
        raise ValueError(f'Cannot index an IntervalIndex of subtype {subtype} with values of dtype {key_dtype}')
    return key_i8