from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.accessor import (
from pandas.core.arrays import (
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.base import (
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
class CombinedDatetimelikeProperties(DatetimeProperties, TimedeltaProperties, PeriodProperties):

    def __new__(cls, data: Series):
        if not isinstance(data, ABCSeries):
            raise TypeError(f'cannot convert an object of type {type(data)} to a datetimelike index')
        orig = data if isinstance(data.dtype, CategoricalDtype) else None
        if orig is not None:
            data = data._constructor(orig.array, name=orig.name, copy=False, dtype=orig._values.categories.dtype, index=orig.index)
        if isinstance(data.dtype, ArrowDtype) and data.dtype.kind in 'Mm':
            return ArrowTemporalProperties(data, orig)
        if lib.is_np_dtype(data.dtype, 'M'):
            return DatetimeProperties(data, orig)
        elif isinstance(data.dtype, DatetimeTZDtype):
            return DatetimeProperties(data, orig)
        elif lib.is_np_dtype(data.dtype, 'm'):
            return TimedeltaProperties(data, orig)
        elif isinstance(data.dtype, PeriodDtype):
            return PeriodProperties(data, orig)
        raise AttributeError('Can only use .dt accessor with datetimelike values')