from __future__ import annotations
import collections
import functools
from typing import (
import numpy as np
from pandas._libs import (
import pandas._libs.groupby as libgroupby
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import (
@final
def agg_series(self, obj: Series, func: Callable, preserve_dtype: bool=False) -> ArrayLike:
    """
        Parameters
        ----------
        obj : Series
        func : function taking a Series and returning a scalar-like
        preserve_dtype : bool
            Whether the aggregation is known to be dtype-preserving.

        Returns
        -------
        np.ndarray or ExtensionArray
        """
    if not isinstance(obj._values, np.ndarray):
        preserve_dtype = True
    result = self._aggregate_series_pure_python(obj, func)
    npvalues = lib.maybe_convert_objects(result, try_float=False)
    if preserve_dtype:
        out = maybe_cast_pointwise_result(npvalues, obj.dtype, numeric_only=True)
    else:
        out = npvalues
    return out