from __future__ import annotations
import copy
from datetime import timedelta
from functools import partial
import inspect
from textwrap import dedent
from typing import (
import numpy as np
from pandas._libs.tslibs import (
import pandas._libs.window.aggregations as window_aggregations
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DataError
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core._numba import executor
from pandas.core.algorithms import factorize
from pandas.core.apply import ResamplerWindowApply
from pandas.core.arrays import ExtensionArray
from pandas.core.base import SelectionMixin
import pandas.core.common as com
from pandas.core.indexers.objects import (
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.util.numba_ import (
from pandas.core.window.common import (
from pandas.core.window.doc import (
from pandas.core.window.numba_ import (
from pandas.core.arrays.datetimelike import dtype_to_unit
def _generate_cython_apply_func(self, args: tuple[Any, ...], kwargs: dict[str, Any], raw: bool | np.bool_, function: Callable[..., Any]) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]:
    from pandas import Series
    window_func = partial(window_aggregations.roll_apply, args=args, kwargs=kwargs, raw=raw, function=function)

    def apply_func(values, begin, end, min_periods, raw=raw):
        if not raw:
            values = Series(values, index=self._on, copy=False)
        return window_func(values, begin, end, min_periods)
    return apply_func