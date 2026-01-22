from __future__ import annotations
import numbers
from typing import (
import numpy as np
from pandas._libs import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import (
from pandas.core.arrays.masked import (
@classmethod
def _get_dtype_mapping(cls) -> Mapping[np.dtype, NumericDtype]:
    raise AbstractMethodError(cls)