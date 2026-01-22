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
def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
    value_left, value_right = self._validate_setitem_value(value)
    if isinstance(self._left, np.ndarray):
        np.putmask(self._left, mask, value_left)
        assert isinstance(self._right, np.ndarray)
        np.putmask(self._right, mask, value_right)
    else:
        self._left._putmask(mask, value_left)
        assert not isinstance(self._right, np.ndarray)
        self._right._putmask(mask, value_right)