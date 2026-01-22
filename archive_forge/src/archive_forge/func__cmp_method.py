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
def _cmp_method(self, other, op):
    pc_func = ARROW_CMP_FUNCS[op.__name__]
    if isinstance(other, (ArrowExtensionArray, np.ndarray, list, BaseMaskedArray)) or isinstance(getattr(other, 'dtype', None), CategoricalDtype):
        result = pc_func(self._pa_array, self._box_pa(other))
    elif is_scalar(other):
        try:
            result = pc_func(self._pa_array, self._box_pa(other))
        except (pa.lib.ArrowNotImplementedError, pa.lib.ArrowInvalid):
            mask = isna(self) | isna(other)
            valid = ~mask
            result = np.zeros(len(self), dtype='bool')
            np_array = np.array(self)
            try:
                result[valid] = op(np_array[valid], other)
            except TypeError:
                result = ops.invalid_comparison(np_array, other, op)
            result = pa.array(result, type=pa.bool_())
            result = pc.if_else(valid, result, None)
    else:
        raise NotImplementedError(f'{op.__name__} not implemented for {type(other)}')
    return ArrowExtensionArray(result)