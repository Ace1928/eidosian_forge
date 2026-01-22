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
class SeriesApply(NDFrameApply):
    obj: Series
    axis: AxisInt = 0
    by_row: Literal[False, 'compat', '_compat']

    def __init__(self, obj: Series, func: AggFuncType, *, convert_dtype: bool | lib.NoDefault=lib.no_default, by_row: Literal[False, 'compat', '_compat']='compat', args, kwargs) -> None:
        if convert_dtype is lib.no_default:
            convert_dtype = True
        else:
            warnings.warn('the convert_dtype parameter is deprecated and will be removed in a future version.  Do ``ser.astype(object).apply()`` instead if you want ``convert_dtype=False``.', FutureWarning, stacklevel=find_stack_level())
        self.convert_dtype = convert_dtype
        super().__init__(obj, func, raw=False, result_type=None, by_row=by_row, args=args, kwargs=kwargs)

    def apply(self) -> DataFrame | Series:
        obj = self.obj
        if len(obj) == 0:
            return self.apply_empty_result()
        if is_list_like(self.func):
            return self.apply_list_or_dict_like()
        if isinstance(self.func, str):
            return self.apply_str()
        if self.by_row == '_compat':
            return self.apply_compat()
        return self.apply_standard()

    def agg(self):
        result = super().agg()
        if result is None:
            obj = self.obj
            func = self.func
            assert callable(func)
            try:
                result = obj.apply(func, args=self.args, **self.kwargs)
            except (ValueError, AttributeError, TypeError):
                result = func(obj, *self.args, **self.kwargs)
            else:
                msg = f'using {func} in {type(obj).__name__}.agg cannot aggregate and has been deprecated. Use {type(obj).__name__}.transform to keep behavior unchanged.'
                warnings.warn(msg, FutureWarning, stacklevel=find_stack_level())
        return result

    def apply_empty_result(self) -> Series:
        obj = self.obj
        return obj._constructor(dtype=obj.dtype, index=obj.index).__finalize__(obj, method='apply')

    def apply_compat(self):
        """compat apply method for funcs in listlikes and dictlikes.

         Used for each callable when giving listlikes and dictlikes of callables to
         apply. Needed for compatibility with Pandas < v2.1.

        .. versionadded:: 2.1.0
        """
        obj = self.obj
        func = self.func
        if callable(func):
            f = com.get_cython_func(func)
            if f and (not self.args) and (not self.kwargs):
                return obj.apply(func, by_row=False)
        try:
            result = obj.apply(func, by_row='compat')
        except (ValueError, AttributeError, TypeError):
            result = obj.apply(func, by_row=False)
        return result

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