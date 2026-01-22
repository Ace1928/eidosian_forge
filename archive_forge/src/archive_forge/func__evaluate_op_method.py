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
def _evaluate_op_method(self, other, op, arrow_funcs):
    pa_type = self._pa_array.type
    other = self._box_pa(other)
    if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type) or pa.types.is_binary(pa_type):
        if op in [operator.add, roperator.radd]:
            sep = pa.scalar('', type=pa_type)
            if op is operator.add:
                result = pc.binary_join_element_wise(self._pa_array, other, sep)
            elif op is roperator.radd:
                result = pc.binary_join_element_wise(other, self._pa_array, sep)
            return type(self)(result)
        elif op in [operator.mul, roperator.rmul]:
            binary = self._pa_array
            integral = other
            if not pa.types.is_integer(integral.type):
                raise TypeError('Can only string multiply by an integer.')
            pa_integral = pc.if_else(pc.less(integral, 0), 0, integral)
            result = pc.binary_repeat(binary, pa_integral)
            return type(self)(result)
    elif (pa.types.is_string(other.type) or pa.types.is_binary(other.type) or pa.types.is_large_string(other.type)) and op in [operator.mul, roperator.rmul]:
        binary = other
        integral = self._pa_array
        if not pa.types.is_integer(integral.type):
            raise TypeError('Can only string multiply by an integer.')
        pa_integral = pc.if_else(pc.less(integral, 0), 0, integral)
        result = pc.binary_repeat(binary, pa_integral)
        return type(self)(result)
    if isinstance(other, pa.Scalar) and pc.is_null(other).as_py() and (op.__name__ in ARROW_LOGICAL_FUNCS):
        other = other.cast(pa_type)
    pc_func = arrow_funcs[op.__name__]
    if pc_func is NotImplemented:
        raise NotImplementedError(f'{op.__name__} not implemented.')
    result = pc_func(self._pa_array, other)
    return type(self)(result)