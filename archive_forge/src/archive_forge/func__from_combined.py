from __future__ import annotations
import operator
from operator import (
import textwrap
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.interval import (
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import IntCastingNaNError
from pandas.util._decorators import Appender
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core.algorithms import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import (
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import check_array_indexer
from pandas.core.ops import (
from_arrays
from_tuples
from_breaks
def _from_combined(self, combined: np.ndarray) -> IntervalArray:
    """
        Create a new IntervalArray with our dtype from a 1D complex128 ndarray.
        """
    nc = combined.view('i8').reshape(-1, 2)
    dtype = self._left.dtype
    if needs_i8_conversion(dtype):
        assert isinstance(self._left, (DatetimeArray, TimedeltaArray))
        new_left = type(self._left)._from_sequence(nc[:, 0], dtype=dtype)
        assert isinstance(self._right, (DatetimeArray, TimedeltaArray))
        new_right = type(self._right)._from_sequence(nc[:, 1], dtype=dtype)
    else:
        assert isinstance(dtype, np.dtype)
        new_left = nc[:, 0].view(dtype)
        new_right = nc[:, 1].view(dtype)
    return self._shallow_copy(left=new_left, right=new_right)