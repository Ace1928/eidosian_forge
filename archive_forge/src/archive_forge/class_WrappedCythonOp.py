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
class WrappedCythonOp:
    """
    Dispatch logic for functions defined in _libs.groupby

    Parameters
    ----------
    kind: str
        Whether the operation is an aggregate or transform.
    how: str
        Operation name, e.g. "mean".
    has_dropped_na: bool
        True precisely when dropna=True and the grouper contains a null value.
    """
    cast_blocklist = frozenset(['any', 'all', 'rank', 'count', 'size', 'idxmin', 'idxmax'])

    def __init__(self, kind: str, how: str, has_dropped_na: bool) -> None:
        self.kind = kind
        self.how = how
        self.has_dropped_na = has_dropped_na
    _CYTHON_FUNCTIONS: dict[str, dict] = {'aggregate': {'any': functools.partial(libgroupby.group_any_all, val_test='any'), 'all': functools.partial(libgroupby.group_any_all, val_test='all'), 'sum': 'group_sum', 'prod': 'group_prod', 'idxmin': functools.partial(libgroupby.group_idxmin_idxmax, name='idxmin'), 'idxmax': functools.partial(libgroupby.group_idxmin_idxmax, name='idxmax'), 'min': 'group_min', 'max': 'group_max', 'mean': 'group_mean', 'median': 'group_median_float64', 'var': 'group_var', 'std': functools.partial(libgroupby.group_var, name='std'), 'sem': functools.partial(libgroupby.group_var, name='sem'), 'skew': 'group_skew', 'first': 'group_nth', 'last': 'group_last', 'ohlc': 'group_ohlc'}, 'transform': {'cumprod': 'group_cumprod', 'cumsum': 'group_cumsum', 'cummin': 'group_cummin', 'cummax': 'group_cummax', 'rank': 'group_rank'}}
    _cython_arity = {'ohlc': 4}

    @classmethod
    def get_kind_from_how(cls, how: str) -> str:
        if how in cls._CYTHON_FUNCTIONS['aggregate']:
            return 'aggregate'
        return 'transform'

    @classmethod
    @functools.cache
    def _get_cython_function(cls, kind: str, how: str, dtype: np.dtype, is_numeric: bool):
        dtype_str = dtype.name
        ftype = cls._CYTHON_FUNCTIONS[kind][how]
        if callable(ftype):
            f = ftype
        else:
            f = getattr(libgroupby, ftype)
        if is_numeric:
            return f
        elif dtype == np.dtype(object):
            if how in ['median', 'cumprod']:
                raise NotImplementedError(f'function is not implemented for this dtype: [how->{how},dtype->{dtype_str}]')
            elif how in ['std', 'sem', 'idxmin', 'idxmax']:
                return f
            elif how == 'skew':
                pass
            elif 'object' not in f.__signatures__:
                raise NotImplementedError(f'function is not implemented for this dtype: [how->{how},dtype->{dtype_str}]')
            return f
        else:
            raise NotImplementedError('This should not be reached. Please report a bug at github.com/pandas-dev/pandas/', dtype)

    def _get_cython_vals(self, values: np.ndarray) -> np.ndarray:
        """
        Cast numeric dtypes to float64 for functions that only support that.

        Parameters
        ----------
        values : np.ndarray

        Returns
        -------
        values : np.ndarray
        """
        how = self.how
        if how in ['median', 'std', 'sem', 'skew']:
            values = ensure_float64(values)
        elif values.dtype.kind in 'iu':
            if how in ['var', 'mean'] or (self.kind == 'transform' and self.has_dropped_na):
                values = ensure_float64(values)
            elif how in ['sum', 'ohlc', 'prod', 'cumsum', 'cumprod']:
                if values.dtype.kind == 'i':
                    values = ensure_int64(values)
                else:
                    values = ensure_uint64(values)
        return values

    def _get_output_shape(self, ngroups: int, values: np.ndarray) -> Shape:
        how = self.how
        kind = self.kind
        arity = self._cython_arity.get(how, 1)
        out_shape: Shape
        if how == 'ohlc':
            out_shape = (ngroups, arity)
        elif arity > 1:
            raise NotImplementedError("arity of more than 1 is not supported for the 'how' argument")
        elif kind == 'transform':
            out_shape = values.shape
        else:
            out_shape = (ngroups,) + values.shape[1:]
        return out_shape

    def _get_out_dtype(self, dtype: np.dtype) -> np.dtype:
        how = self.how
        if how == 'rank':
            out_dtype = 'float64'
        elif how in ['idxmin', 'idxmax']:
            out_dtype = 'intp'
        elif dtype.kind in 'iufcb':
            out_dtype = f'{dtype.kind}{dtype.itemsize}'
        else:
            out_dtype = 'object'
        return np.dtype(out_dtype)

    def _get_result_dtype(self, dtype: np.dtype) -> np.dtype:
        """
        Get the desired dtype of a result based on the
        input dtype and how it was computed.

        Parameters
        ----------
        dtype : np.dtype

        Returns
        -------
        np.dtype
            The desired dtype of the result.
        """
        how = self.how
        if how in ['sum', 'cumsum', 'sum', 'prod', 'cumprod']:
            if dtype == np.dtype(bool):
                return np.dtype(np.int64)
        elif how in ['mean', 'median', 'var', 'std', 'sem']:
            if dtype.kind in 'fc':
                return dtype
            elif dtype.kind in 'iub':
                return np.dtype(np.float64)
        return dtype

    @final
    def _cython_op_ndim_compat(self, values: np.ndarray, *, min_count: int, ngroups: int, comp_ids: np.ndarray, mask: npt.NDArray[np.bool_] | None=None, result_mask: npt.NDArray[np.bool_] | None=None, **kwargs) -> np.ndarray:
        if values.ndim == 1:
            values2d = values[None, :]
            if mask is not None:
                mask = mask[None, :]
            if result_mask is not None:
                result_mask = result_mask[None, :]
            res = self._call_cython_op(values2d, min_count=min_count, ngroups=ngroups, comp_ids=comp_ids, mask=mask, result_mask=result_mask, **kwargs)
            if res.shape[0] == 1:
                return res[0]
            return res.T
        return self._call_cython_op(values, min_count=min_count, ngroups=ngroups, comp_ids=comp_ids, mask=mask, result_mask=result_mask, **kwargs)

    @final
    def _call_cython_op(self, values: np.ndarray, *, min_count: int, ngroups: int, comp_ids: np.ndarray, mask: npt.NDArray[np.bool_] | None, result_mask: npt.NDArray[np.bool_] | None, **kwargs) -> np.ndarray:
        orig_values = values
        dtype = values.dtype
        is_numeric = dtype.kind in 'iufcb'
        is_datetimelike = dtype.kind in 'mM'
        if is_datetimelike:
            values = values.view('int64')
            is_numeric = True
        elif dtype.kind == 'b':
            values = values.view('uint8')
        if values.dtype == 'float16':
            values = values.astype(np.float32)
        if self.how in ['any', 'all']:
            if mask is None:
                mask = isna(values)
            if dtype == object:
                if kwargs['skipna']:
                    if mask.any():
                        values = values.copy()
                        values[mask] = True
            values = values.astype(bool, copy=False).view(np.int8)
            is_numeric = True
        values = values.T
        if mask is not None:
            mask = mask.T
            if result_mask is not None:
                result_mask = result_mask.T
        out_shape = self._get_output_shape(ngroups, values)
        func = self._get_cython_function(self.kind, self.how, values.dtype, is_numeric)
        values = self._get_cython_vals(values)
        out_dtype = self._get_out_dtype(values.dtype)
        result = maybe_fill(np.empty(out_shape, dtype=out_dtype))
        if self.kind == 'aggregate':
            counts = np.zeros(ngroups, dtype=np.int64)
            if self.how in ['idxmin', 'idxmax', 'min', 'max', 'mean', 'last', 'first', 'sum']:
                func(out=result, counts=counts, values=values, labels=comp_ids, min_count=min_count, mask=mask, result_mask=result_mask, is_datetimelike=is_datetimelike, **kwargs)
            elif self.how in ['sem', 'std', 'var', 'ohlc', 'prod', 'median']:
                if self.how in ['std', 'sem']:
                    kwargs['is_datetimelike'] = is_datetimelike
                func(result, counts, values, comp_ids, min_count=min_count, mask=mask, result_mask=result_mask, **kwargs)
            elif self.how in ['any', 'all']:
                func(out=result, values=values, labels=comp_ids, mask=mask, result_mask=result_mask, **kwargs)
                result = result.astype(bool, copy=False)
            elif self.how in ['skew']:
                func(out=result, counts=counts, values=values, labels=comp_ids, mask=mask, result_mask=result_mask, **kwargs)
                if dtype == object:
                    result = result.astype(object)
            else:
                raise NotImplementedError(f'{self.how} is not implemented')
        else:
            if self.how != 'rank':
                kwargs['result_mask'] = result_mask
            func(out=result, values=values, labels=comp_ids, ngroups=ngroups, is_datetimelike=is_datetimelike, mask=mask, **kwargs)
        if self.kind == 'aggregate' and self.how not in ['idxmin', 'idxmax']:
            if result.dtype.kind in 'iu' and (not is_datetimelike):
                cutoff = max(0 if self.how in ['sum', 'prod'] else 1, min_count)
                empty_groups = counts < cutoff
                if empty_groups.any():
                    if result_mask is not None:
                        assert result_mask[empty_groups].all()
                    else:
                        result = result.astype('float64')
                        result[empty_groups] = np.nan
        result = result.T
        if self.how not in self.cast_blocklist:
            res_dtype = self._get_result_dtype(orig_values.dtype)
            op_result = maybe_downcast_to_dtype(result, res_dtype)
        else:
            op_result = result
        return op_result

    @final
    def _validate_axis(self, axis: AxisInt, values: ArrayLike) -> None:
        if values.ndim > 2:
            raise NotImplementedError('number of dimensions is currently limited to 2')
        if values.ndim == 2:
            assert axis == 1, axis
        elif not is_1d_only_ea_dtype(values.dtype):
            assert axis == 0

    @final
    def cython_operation(self, *, values: ArrayLike, axis: AxisInt, min_count: int=-1, comp_ids: np.ndarray, ngroups: int, **kwargs) -> ArrayLike:
        """
        Call our cython function, with appropriate pre- and post- processing.
        """
        self._validate_axis(axis, values)
        if not isinstance(values, np.ndarray):
            return values._groupby_op(how=self.how, has_dropped_na=self.has_dropped_na, min_count=min_count, ngroups=ngroups, ids=comp_ids, **kwargs)
        return self._cython_op_ndim_compat(values, min_count=min_count, ngroups=ngroups, comp_ids=comp_ids, mask=None, **kwargs)