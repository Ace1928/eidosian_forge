from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import is_supported_dtype
from pandas.compat.numpy import function as nv
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.strings.object_array import ObjectStringArrayMixin
def _wrap_ndarray_result(self, result: np.ndarray):
    if result.dtype.kind == 'm' and is_supported_dtype(result.dtype):
        from pandas.core.arrays import TimedeltaArray
        return TimedeltaArray._simple_new(result, dtype=result.dtype)
    return type(self)(result)