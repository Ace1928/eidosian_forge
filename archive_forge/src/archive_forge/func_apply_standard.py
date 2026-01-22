from __future__ import annotations
import abc
from collections import defaultdict
import functools
from functools import partial
import inspect
from typing import (
import warnings
import numpy as np
from pandas._config import option_context
from pandas._libs import lib
from pandas._libs.internals import BlockValuesRefs
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SpecificationError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import is_nested_object
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core._numba.executor import generate_apply_looper
import pandas.core.common as com
from pandas.core.construction import ensure_wrapped_if_datetimelike
def apply_standard(self) -> DataFrame | Series:
    func = cast(Callable, self.func)
    obj = self.obj
    if isinstance(func, np.ufunc):
        with np.errstate(all='ignore'):
            return func(obj, *self.args, **self.kwargs)
    elif not self.by_row:
        return func(obj, *self.args, **self.kwargs)
    if self.args or self.kwargs:

        def curried(x):
            return func(x, *self.args, **self.kwargs)
    else:
        curried = func
    action = 'ignore' if isinstance(obj.dtype, CategoricalDtype) else None
    mapped = obj._map_values(mapper=curried, na_action=action, convert=self.convert_dtype)
    if len(mapped) and isinstance(mapped[0], ABCSeries):
        return obj._constructor_expanddim(list(mapped), index=obj.index)
    else:
        return obj._constructor(mapped, index=obj.index).__finalize__(obj, method='apply')