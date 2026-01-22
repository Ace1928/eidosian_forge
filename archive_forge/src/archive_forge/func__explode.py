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
def _explode(self):
    """
        See Series.explode.__doc__.
        """
    if not pa.types.is_list(self.dtype.pyarrow_dtype):
        return super()._explode()
    values = self
    counts = pa.compute.list_value_length(values._pa_array)
    counts = counts.fill_null(1).to_numpy()
    fill_value = pa.scalar([None], type=self._pa_array.type)
    mask = counts == 0
    if mask.any():
        values = values.copy()
        values[mask] = fill_value
        counts = counts.copy()
        counts[mask] = 1
    values = values.fillna(fill_value)
    values = type(self)(pa.compute.list_flatten(values._pa_array))
    return (values, counts)