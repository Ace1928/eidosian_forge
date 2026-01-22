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
def _round_temporally(self, method: Literal['ceil', 'floor', 'round'], freq, ambiguous: TimeAmbiguous='raise', nonexistent: TimeNonexistent='raise'):
    if ambiguous != 'raise':
        raise NotImplementedError('ambiguous is not supported.')
    if nonexistent != 'raise':
        raise NotImplementedError('nonexistent is not supported.')
    offset = to_offset(freq)
    if offset is None:
        raise ValueError(f'Must specify a valid frequency: {freq}')
    pa_supported_unit = {'Y': 'year', 'YS': 'year', 'Q': 'quarter', 'QS': 'quarter', 'M': 'month', 'MS': 'month', 'W': 'week', 'D': 'day', 'h': 'hour', 'min': 'minute', 's': 'second', 'ms': 'millisecond', 'us': 'microsecond', 'ns': 'nanosecond'}
    unit = pa_supported_unit.get(offset._prefix, None)
    if unit is None:
        raise ValueError(f'freq={freq!r} is not supported')
    multiple = offset.n
    rounding_method = getattr(pc, f'{method}_temporal')
    return type(self)(rounding_method(self._pa_array, multiple=multiple, unit=unit))