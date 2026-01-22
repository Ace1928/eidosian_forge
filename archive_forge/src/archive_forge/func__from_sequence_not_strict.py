from __future__ import annotations
from datetime import (
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_inclusive
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.tseries.frequencies import get_period_alias
from pandas.tseries.offsets import (
@classmethod
def _from_sequence_not_strict(cls, data, *, dtype=None, copy: bool=False, tz=lib.no_default, freq: str | BaseOffset | lib.NoDefault | None=lib.no_default, dayfirst: bool=False, yearfirst: bool=False, ambiguous: TimeAmbiguous='raise') -> Self:
    """
        A non-strict version of _from_sequence, called from DatetimeIndex.__new__.
        """
    explicit_tz_none = tz is None
    if tz is lib.no_default:
        tz = None
    else:
        tz = timezones.maybe_get_tz(tz)
    dtype = _validate_dt64_dtype(dtype)
    tz = _validate_tz_from_dtype(dtype, tz, explicit_tz_none)
    unit = None
    if dtype is not None:
        unit = dtl.dtype_to_unit(dtype)
    data, copy = dtl.ensure_arraylike_for_datetimelike(data, copy, cls_name='DatetimeArray')
    inferred_freq = None
    if isinstance(data, DatetimeArray):
        inferred_freq = data.freq
    subarr, tz = _sequence_to_dt64(data, copy=copy, tz=tz, dayfirst=dayfirst, yearfirst=yearfirst, ambiguous=ambiguous, out_unit=unit)
    _validate_tz_from_dtype(dtype, tz, explicit_tz_none)
    if tz is not None and explicit_tz_none:
        raise ValueError("Passed data is timezone-aware, incompatible with 'tz=None'. Use obj.tz_localize(None) instead.")
    data_unit = np.datetime_data(subarr.dtype)[0]
    data_dtype = tz_to_dtype(tz, data_unit)
    result = cls._simple_new(subarr, freq=inferred_freq, dtype=data_dtype)
    if unit is not None and unit != result.unit:
        result = result.as_unit(unit)
    validate_kwds = {'ambiguous': ambiguous}
    result._maybe_pin_freq(freq, validate_kwds)
    return result