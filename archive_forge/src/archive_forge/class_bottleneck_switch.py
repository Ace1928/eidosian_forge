from __future__ import annotations
import functools
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
class bottleneck_switch:

    def __init__(self, name=None, **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs

    def __call__(self, alt: F) -> F:
        bn_name = self.name or alt.__name__
        try:
            bn_func = getattr(bn, bn_name)
        except (AttributeError, NameError):
            bn_func = None

        @functools.wraps(alt)
        def f(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, **kwds):
            if len(self.kwargs) > 0:
                for k, v in self.kwargs.items():
                    if k not in kwds:
                        kwds[k] = v
            if values.size == 0 and kwds.get('min_count') is None:
                return _na_for_min_count(values, axis)
            if _USE_BOTTLENECK and skipna and _bn_ok_dtype(values.dtype, bn_name):
                if kwds.get('mask', None) is None:
                    kwds.pop('mask', None)
                    result = bn_func(values, axis=axis, **kwds)
                    if _has_infs(result):
                        result = alt(values, axis=axis, skipna=skipna, **kwds)
                else:
                    result = alt(values, axis=axis, skipna=skipna, **kwds)
            else:
                result = alt(values, axis=axis, skipna=skipna, **kwds)
            return result
        return cast(F, f)