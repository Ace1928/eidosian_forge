from __future__ import annotations
from typing import ClassVar
import numpy as np
from pandas.core.dtypes.base import register_extension_dtype
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.arrays.numeric import (
@register_extension_dtype
class Int32Dtype(IntegerDtype):
    type = np.int32
    name: ClassVar[str] = 'Int32'
    __doc__ = _dtype_docstring.format(dtype='int32')