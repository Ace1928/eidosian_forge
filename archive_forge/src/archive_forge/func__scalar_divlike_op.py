from __future__ import annotations
from datetime import timedelta
import operator
from typing import (
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.fields import (
from pandas._libs.tslibs.timedeltas import (
from pandas.compat.numpy import function as nv
from pandas.util._validators import validate_endpoints
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.core.ops.common import unpack_zerodim_and_defer
import textwrap
def _scalar_divlike_op(self, other, op):
    """
        Shared logic for __truediv__, __rtruediv__, __floordiv__, __rfloordiv__
        with scalar 'other'.
        """
    if isinstance(other, self._recognized_scalars):
        other = Timedelta(other)
        if cast('Timedelta | NaTType', other) is NaT:
            res = np.empty(self.shape, dtype=np.float64)
            res.fill(np.nan)
            return res
        return op(self._ndarray, other)
    else:
        if op in [roperator.rtruediv, roperator.rfloordiv]:
            raise TypeError(f'Cannot divide {type(other).__name__} by {type(self).__name__}')
        result = op(self._ndarray, other)
        freq = None
        if self.freq is not None:
            freq = self.freq / other
            if freq.nanos == 0 and self.freq.nanos != 0:
                freq = None
        return type(self)._simple_new(result, dtype=result.dtype, freq=freq)