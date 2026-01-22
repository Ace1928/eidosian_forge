from __future__ import annotations
import functools
import operator
import re
import textwrap
from typing import (
import unicodedata
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas.compat import (
from pandas.util._decorators import doc
from pandas.util._validators import validate_fillna_kwargs
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.algorithms import map_array
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import (
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.indexers import (
from pandas.core.strings.base import BaseStringArrayMethods
from pandas.io._util import _arrow_dtype_mapping
from pandas.tseries.frequencies import to_offset
@classmethod
def _box_pa_scalar(cls, value, pa_type: pa.DataType | None=None) -> pa.Scalar:
    """
        Box value into a pyarrow Scalar.

        Parameters
        ----------
        value : any
        pa_type : pa.DataType | None

        Returns
        -------
        pa.Scalar
        """
    if isinstance(value, pa.Scalar):
        pa_scalar = value
    elif isna(value):
        pa_scalar = pa.scalar(None, type=pa_type)
    else:
        if isinstance(value, Timedelta):
            if pa_type is None:
                pa_type = pa.duration(value.unit)
            elif value.unit != pa_type.unit:
                value = value.as_unit(pa_type.unit)
            value = value._value
        elif isinstance(value, Timestamp):
            if pa_type is None:
                pa_type = pa.timestamp(value.unit, tz=value.tz)
            elif value.unit != pa_type.unit:
                value = value.as_unit(pa_type.unit)
            value = value._value
        pa_scalar = pa.scalar(value, type=pa_type, from_pandas=True)
    if pa_type is not None and pa_scalar.type != pa_type:
        pa_scalar = pa_scalar.cast(pa_type)
    return pa_scalar