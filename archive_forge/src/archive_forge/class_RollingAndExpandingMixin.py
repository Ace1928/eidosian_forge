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
class RollingAndExpandingMixin(BaseWindow):

    def count(self, numeric_only: bool=False):
        window_func = window_aggregations.roll_sum
        return self._apply(window_func, name='count', numeric_only=numeric_only)

    def apply(self, func: Callable[..., Any], raw: bool=False, engine: Literal['cython', 'numba'] | None=None, engine_kwargs: dict[str, bool] | None=None, args: tuple[Any, ...] | None=None, kwargs: dict[str, Any] | None=None):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if not is_bool(raw):
            raise ValueError('raw parameter must be `True` or `False`')
        numba_args: tuple[Any, ...] = ()
        if maybe_use_numba(engine):
            if raw is False:
                raise ValueError('raw must be `True` when using the numba engine')
            numba_args = args
            if self.method == 'single':
                apply_func = generate_numba_apply_func(func, **get_jit_arguments(engine_kwargs, kwargs))
            else:
                apply_func = generate_numba_table_func(func, **get_jit_arguments(engine_kwargs, kwargs))
        elif engine in ('cython', None):
            if engine_kwargs is not None:
                raise ValueError('cython engine does not accept engine_kwargs')
            apply_func = self._generate_cython_apply_func(args, kwargs, raw, func)
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")
        return self._apply(apply_func, name='apply', numba_args=numba_args)

    def _generate_cython_apply_func(self, args: tuple[Any, ...], kwargs: dict[str, Any], raw: bool | np.bool_, function: Callable[..., Any]) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]:
        from pandas import Series
        window_func = partial(window_aggregations.roll_apply, args=args, kwargs=kwargs, raw=raw, function=function)

        def apply_func(values, begin, end, min_periods, raw=raw):
            if not raw:
                values = Series(values, index=self._on, copy=False)
            return window_func(values, begin, end, min_periods)
        return apply_func

    def sum(self, numeric_only: bool=False, engine: Literal['cython', 'numba'] | None=None, engine_kwargs: dict[str, bool] | None=None):
        if maybe_use_numba(engine):
            if self.method == 'table':
                func = generate_manual_numpy_nan_agg_with_axis(np.nansum)
                return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
            else:
                from pandas.core._numba.kernels import sliding_sum
                return self._numba_apply(sliding_sum, engine_kwargs)
        window_func = window_aggregations.roll_sum
        return self._apply(window_func, name='sum', numeric_only=numeric_only)

    def max(self, numeric_only: bool=False, engine: Literal['cython', 'numba'] | None=None, engine_kwargs: dict[str, bool] | None=None):
        if maybe_use_numba(engine):
            if self.method == 'table':
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmax)
                return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
            else:
                from pandas.core._numba.kernels import sliding_min_max
                return self._numba_apply(sliding_min_max, engine_kwargs, is_max=True)
        window_func = window_aggregations.roll_max
        return self._apply(window_func, name='max', numeric_only=numeric_only)

    def min(self, numeric_only: bool=False, engine: Literal['cython', 'numba'] | None=None, engine_kwargs: dict[str, bool] | None=None):
        if maybe_use_numba(engine):
            if self.method == 'table':
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmin)
                return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
            else:
                from pandas.core._numba.kernels import sliding_min_max
                return self._numba_apply(sliding_min_max, engine_kwargs, is_max=False)
        window_func = window_aggregations.roll_min
        return self._apply(window_func, name='min', numeric_only=numeric_only)

    def mean(self, numeric_only: bool=False, engine: Literal['cython', 'numba'] | None=None, engine_kwargs: dict[str, bool] | None=None):
        if maybe_use_numba(engine):
            if self.method == 'table':
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmean)
                return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
            else:
                from pandas.core._numba.kernels import sliding_mean
                return self._numba_apply(sliding_mean, engine_kwargs)
        window_func = window_aggregations.roll_mean
        return self._apply(window_func, name='mean', numeric_only=numeric_only)

    def median(self, numeric_only: bool=False, engine: Literal['cython', 'numba'] | None=None, engine_kwargs: dict[str, bool] | None=None):
        if maybe_use_numba(engine):
            if self.method == 'table':
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmedian)
            else:
                func = np.nanmedian
            return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
        window_func = window_aggregations.roll_median_c
        return self._apply(window_func, name='median', numeric_only=numeric_only)

    def std(self, ddof: int=1, numeric_only: bool=False, engine: Literal['cython', 'numba'] | None=None, engine_kwargs: dict[str, bool] | None=None):
        if maybe_use_numba(engine):
            if self.method == 'table':
                raise NotImplementedError("std not supported with method='table'")
            from pandas.core._numba.kernels import sliding_var
            return zsqrt(self._numba_apply(sliding_var, engine_kwargs, ddof=ddof))
        window_func = window_aggregations.roll_var

        def zsqrt_func(values, begin, end, min_periods):
            return zsqrt(window_func(values, begin, end, min_periods, ddof=ddof))
        return self._apply(zsqrt_func, name='std', numeric_only=numeric_only)

    def var(self, ddof: int=1, numeric_only: bool=False, engine: Literal['cython', 'numba'] | None=None, engine_kwargs: dict[str, bool] | None=None):
        if maybe_use_numba(engine):
            if self.method == 'table':
                raise NotImplementedError("var not supported with method='table'")
            from pandas.core._numba.kernels import sliding_var
            return self._numba_apply(sliding_var, engine_kwargs, ddof=ddof)
        window_func = partial(window_aggregations.roll_var, ddof=ddof)
        return self._apply(window_func, name='var', numeric_only=numeric_only)

    def skew(self, numeric_only: bool=False):
        window_func = window_aggregations.roll_skew
        return self._apply(window_func, name='skew', numeric_only=numeric_only)

    def sem(self, ddof: int=1, numeric_only: bool=False):
        self._validate_numeric_only('sem', numeric_only)
        return self.std(numeric_only=numeric_only) / (self.count(numeric_only=numeric_only) - ddof).pow(0.5)

    def kurt(self, numeric_only: bool=False):
        window_func = window_aggregations.roll_kurt
        return self._apply(window_func, name='kurt', numeric_only=numeric_only)

    def quantile(self, q: float, interpolation: QuantileInterpolation='linear', numeric_only: bool=False):
        if q == 1.0:
            window_func = window_aggregations.roll_max
        elif q == 0.0:
            window_func = window_aggregations.roll_min
        else:
            window_func = partial(window_aggregations.roll_quantile, quantile=q, interpolation=interpolation)
        return self._apply(window_func, name='quantile', numeric_only=numeric_only)

    def rank(self, method: WindowingRankType='average', ascending: bool=True, pct: bool=False, numeric_only: bool=False):
        window_func = partial(window_aggregations.roll_rank, method=method, ascending=ascending, percentile=pct)
        return self._apply(window_func, name='rank', numeric_only=numeric_only)

    def cov(self, other: DataFrame | Series | None=None, pairwise: bool | None=None, ddof: int=1, numeric_only: bool=False):
        if self.step is not None:
            raise NotImplementedError('step not implemented for cov')
        self._validate_numeric_only('cov', numeric_only)
        from pandas import Series

        def cov_func(x, y):
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = self.min_periods if self.min_periods is not None else window_indexer.window_size
            start, end = window_indexer.get_window_bounds(num_values=len(x_array), min_periods=min_periods, center=self.center, closed=self.closed, step=self.step)
            self._check_window_bounds(start, end, len(x_array))
            with np.errstate(all='ignore'):
                mean_x_y = window_aggregations.roll_mean(x_array * y_array, start, end, min_periods)
                mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
                mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
                count_x_y = window_aggregations.roll_sum(notna(x_array + y_array).astype(np.float64), start, end, 0)
                result = (mean_x_y - mean_x * mean_y) * (count_x_y / (count_x_y - ddof))
            return Series(result, index=x.index, name=x.name, copy=False)
        return self._apply_pairwise(self._selected_obj, other, pairwise, cov_func, numeric_only)

    def corr(self, other: DataFrame | Series | None=None, pairwise: bool | None=None, ddof: int=1, numeric_only: bool=False):
        if self.step is not None:
            raise NotImplementedError('step not implemented for corr')
        self._validate_numeric_only('corr', numeric_only)
        from pandas import Series

        def corr_func(x, y):
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = self.min_periods if self.min_periods is not None else window_indexer.window_size
            start, end = window_indexer.get_window_bounds(num_values=len(x_array), min_periods=min_periods, center=self.center, closed=self.closed, step=self.step)
            self._check_window_bounds(start, end, len(x_array))
            with np.errstate(all='ignore'):
                mean_x_y = window_aggregations.roll_mean(x_array * y_array, start, end, min_periods)
                mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
                mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
                count_x_y = window_aggregations.roll_sum(notna(x_array + y_array).astype(np.float64), start, end, 0)
                x_var = window_aggregations.roll_var(x_array, start, end, min_periods, ddof)
                y_var = window_aggregations.roll_var(y_array, start, end, min_periods, ddof)
                numerator = (mean_x_y - mean_x * mean_y) * (count_x_y / (count_x_y - ddof))
                denominator = (x_var * y_var) ** 0.5
                result = numerator / denominator
            return Series(result, index=x.index, name=x.name, copy=False)
        return self._apply_pairwise(self._selected_obj, other, pairwise, corr_func, numeric_only)