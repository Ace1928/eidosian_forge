from __future__ import annotations
from datetime import (
from functools import wraps
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
from pandas._libs.tslibs.fields import (
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas._libs.tslibs.timedeltas import get_unit_for_round
from pandas._libs.tslibs.timestamps import integer_op_not_supported
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.algorithms import (
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import (
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.integer import IntegerArray
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.ops.invalid import (
from pandas.tseries import frequencies
@final
def _add_datetimelike_scalar(self, other) -> DatetimeArray:
    if not lib.is_np_dtype(self.dtype, 'm'):
        raise TypeError(f'cannot add {type(self).__name__} and {type(other).__name__}')
    self = cast('TimedeltaArray', self)
    from pandas.core.arrays import DatetimeArray
    from pandas.core.arrays.datetimes import tz_to_dtype
    assert other is not NaT
    if isna(other):
        result = self._ndarray + NaT.to_datetime64().astype(f'M8[{self.unit}]')
        return DatetimeArray._simple_new(result, dtype=result.dtype)
    other = Timestamp(other)
    self, other = self._ensure_matching_resos(other)
    self = cast('TimedeltaArray', self)
    other_i8, o_mask = self._get_i8_values_and_mask(other)
    result = add_overflowsafe(self.asi8, np.asarray(other_i8, dtype='i8'))
    res_values = result.view(f'M8[{self.unit}]')
    dtype = tz_to_dtype(tz=other.tz, unit=self.unit)
    res_values = result.view(f'M8[{self.unit}]')
    new_freq = self._get_arithmetic_result_freq(other)
    return DatetimeArray._simple_new(res_values, dtype=dtype, freq=new_freq)