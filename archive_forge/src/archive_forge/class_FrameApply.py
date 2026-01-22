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
class FrameApply(NDFrameApply):
    obj: DataFrame

    def __init__(self, obj: AggObjType, func: AggFuncType, raw: bool, result_type: str | None, *, by_row: Literal[False, 'compat']=False, engine: str='python', engine_kwargs: dict[str, bool] | None=None, args, kwargs) -> None:
        if by_row is not False and by_row != 'compat':
            raise ValueError(f'by_row={by_row} not allowed')
        super().__init__(obj, func, raw, result_type, by_row=by_row, engine=engine, engine_kwargs=engine_kwargs, args=args, kwargs=kwargs)

    @property
    @abc.abstractmethod
    def result_index(self) -> Index:
        pass

    @property
    @abc.abstractmethod
    def result_columns(self) -> Index:
        pass

    @property
    @abc.abstractmethod
    def series_generator(self) -> Generator[Series, None, None]:
        pass

    @staticmethod
    @functools.cache
    @abc.abstractmethod
    def generate_numba_apply_func(func, nogil=True, nopython=True, parallel=False) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        pass

    @abc.abstractmethod
    def apply_with_numba(self):
        pass

    def validate_values_for_numba(self):
        for colname, dtype in self.obj.dtypes.items():
            if not is_numeric_dtype(dtype):
                raise ValueError(f"Column {colname} must have a numeric dtype. Found '{dtype}' instead")
            if is_extension_array_dtype(dtype):
                raise ValueError(f'Column {colname} is backed by an extension array, which is not supported by the numba engine.')

    @abc.abstractmethod
    def wrap_results_for_axis(self, results: ResType, res_index: Index) -> DataFrame | Series:
        pass

    @property
    def res_columns(self) -> Index:
        return self.result_columns

    @property
    def columns(self) -> Index:
        return self.obj.columns

    @cache_readonly
    def values(self):
        return self.obj.values

    def apply(self) -> DataFrame | Series:
        """compute the results"""
        if is_list_like(self.func):
            if self.engine == 'numba':
                raise NotImplementedError("the 'numba' engine doesn't support lists of callables yet")
            return self.apply_list_or_dict_like()
        if len(self.columns) == 0 and len(self.index) == 0:
            return self.apply_empty_result()
        if isinstance(self.func, str):
            if self.engine == 'numba':
                raise NotImplementedError("the 'numba' engine doesn't support using a string as the callable function")
            return self.apply_str()
        elif isinstance(self.func, np.ufunc):
            if self.engine == 'numba':
                raise NotImplementedError("the 'numba' engine doesn't support using a numpy ufunc as the callable function")
            with np.errstate(all='ignore'):
                results = self.obj._mgr.apply('apply', func=self.func)
            return self.obj._constructor_from_mgr(results, axes=results.axes)
        if self.result_type == 'broadcast':
            if self.engine == 'numba':
                raise NotImplementedError("the 'numba' engine doesn't support result_type='broadcast'")
            return self.apply_broadcast(self.obj)
        elif not all(self.obj.shape):
            return self.apply_empty_result()
        elif self.raw:
            return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
        return self.apply_standard()

    def agg(self):
        obj = self.obj
        axis = self.axis
        self.obj = self.obj if self.axis == 0 else self.obj.T
        self.axis = 0
        result = None
        try:
            result = super().agg()
        finally:
            self.obj = obj
            self.axis = axis
        if axis == 1:
            result = result.T if result is not None else result
        if result is None:
            result = self.obj.apply(self.func, axis, args=self.args, **self.kwargs)
        return result

    def apply_empty_result(self):
        """
        we have an empty result; at least 1 axis is 0

        we will try to apply the function to an empty
        series in order to see if this is a reduction function
        """
        assert callable(self.func)
        if self.result_type not in ['reduce', None]:
            return self.obj.copy()
        should_reduce = self.result_type == 'reduce'
        from pandas import Series
        if not should_reduce:
            try:
                if self.axis == 0:
                    r = self.func(Series([], dtype=np.float64), *self.args, **self.kwargs)
                else:
                    r = self.func(Series(index=self.columns, dtype=np.float64), *self.args, **self.kwargs)
            except Exception:
                pass
            else:
                should_reduce = not isinstance(r, Series)
        if should_reduce:
            if len(self.agg_axis):
                r = self.func(Series([], dtype=np.float64), *self.args, **self.kwargs)
            else:
                r = np.nan
            return self.obj._constructor_sliced(r, index=self.agg_axis)
        else:
            return self.obj.copy()

    def apply_raw(self, engine='python', engine_kwargs=None):
        """apply to the values as a numpy array"""

        def wrap_function(func):
            """
            Wrap user supplied function to work around numpy issue.

            see https://github.com/numpy/numpy/issues/8352
            """

            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if isinstance(result, str):
                    result = np.array(result, dtype=object)
                return result
            return wrapper
        if engine == 'numba':
            engine_kwargs = {} if engine_kwargs is None else engine_kwargs
            nb_looper = generate_apply_looper(self.func, **engine_kwargs)
            result = nb_looper(self.values, self.axis)
            result = np.squeeze(result)
        else:
            result = np.apply_along_axis(wrap_function(self.func), self.axis, self.values, *self.args, **self.kwargs)
        if result.ndim == 2:
            return self.obj._constructor(result, index=self.index, columns=self.columns)
        else:
            return self.obj._constructor_sliced(result, index=self.agg_axis)

    def apply_broadcast(self, target: DataFrame) -> DataFrame:
        assert callable(self.func)
        result_values = np.empty_like(target.values)
        result_compare = target.shape[0]
        for i, col in enumerate(target.columns):
            res = self.func(target[col], *self.args, **self.kwargs)
            ares = np.asarray(res).ndim
            if ares > 1:
                raise ValueError('too many dims to broadcast')
            if ares == 1:
                if result_compare != len(res):
                    raise ValueError('cannot broadcast result')
            result_values[:, i] = res
        result = self.obj._constructor(result_values, index=target.index, columns=target.columns)
        return result

    def apply_standard(self):
        if self.engine == 'python':
            results, res_index = self.apply_series_generator()
        else:
            results, res_index = self.apply_series_numba()
        return self.wrap_results(results, res_index)

    def apply_series_generator(self) -> tuple[ResType, Index]:
        assert callable(self.func)
        series_gen = self.series_generator
        res_index = self.result_index
        results = {}
        with option_context('mode.chained_assignment', None):
            for i, v in enumerate(series_gen):
                results[i] = self.func(v, *self.args, **self.kwargs)
                if isinstance(results[i], ABCSeries):
                    results[i] = results[i].copy(deep=False)
        return (results, res_index)

    def apply_series_numba(self):
        if self.engine_kwargs.get('parallel', False):
            raise NotImplementedError("Parallel apply is not supported when raw=False and engine='numba'")
        if not self.obj.index.is_unique or not self.columns.is_unique:
            raise NotImplementedError("The index/columns must be unique when raw=False and engine='numba'")
        self.validate_values_for_numba()
        results = self.apply_with_numba()
        return (results, self.result_index)

    def wrap_results(self, results: ResType, res_index: Index) -> DataFrame | Series:
        from pandas import Series
        if len(results) > 0 and 0 in results and is_sequence(results[0]):
            return self.wrap_results_for_axis(results, res_index)
        constructor_sliced = self.obj._constructor_sliced
        if len(results) == 0 and constructor_sliced is Series:
            result = constructor_sliced(results, dtype=np.float64)
        else:
            result = constructor_sliced(results)
        result.index = res_index
        return result

    def apply_str(self) -> DataFrame | Series:
        if self.func == 'size':
            obj = self.obj
            value = obj.shape[self.axis]
            return obj._constructor_sliced(value, index=self.agg_axis)
        return super().apply_str()