from __future__ import annotations
import collections
import itertools as it
import operator
import uuid
import warnings
from functools import partial, wraps
from numbers import Integral
import numpy as np
import pandas as pd
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.dataframe._compat import (
from dask.dataframe.core import (
from dask.dataframe.dispatch import grouper_dispatch
from dask.dataframe.methods import concat, drop_columns
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import (
class _GroupBy:
    """Superclass for DataFrameGroupBy and SeriesGroupBy

    Parameters
    ----------

    obj: DataFrame or Series
        DataFrame or Series to be grouped
    by: str, list or Series
        The key for grouping
    slice: str, list
        The slice keys applied to GroupBy result
    group_keys: bool | None
        Passed to pandas.DataFrame.groupby()
    dropna: bool
        Whether to drop null values from groupby index
    sort: bool
        Passed along to aggregation methods. If allowed,
        the output aggregation will have sorted keys.
    observed: bool, default False
        This only applies if any of the groupers are Categoricals.
        If True: only show observed values for categorical groupers.
        If False: show all values for categorical groupers.
    """

    def __init__(self, df, by=None, slice=None, group_keys=GROUP_KEYS_DEFAULT, dropna=None, sort=True, observed=None):
        by_ = by if isinstance(by, (tuple, list)) else [by]
        if any((isinstance(key, pd.Grouper) for key in by_)):
            raise NotImplementedError('pd.Grouper is currently not supported by Dask.')
        self._slice = slice
        projection = None
        if np.isscalar(self._slice) or isinstance(self._slice, (str, list, tuple)) or ((is_index_like(self._slice) or is_series_like(self._slice)) and (not is_dask_collection(self._slice))):
            projection = set(by_).union({self._slice} if np.isscalar(self._slice) or isinstance(self._slice, str) else self._slice)
            projection = [c for c in df.columns if c in projection]
        assert isinstance(df, (DataFrame, Series))
        self.group_keys = group_keys
        self.obj = df[projection] if projection else df
        self.by = _normalize_by(df, by)
        self.sort = sort
        partitions_aligned = all((item.npartitions == df.npartitions if isinstance(item, Series) else True for item in (self.by if isinstance(self.by, (tuple, list)) else [self.by])))
        if not partitions_aligned:
            raise NotImplementedError("The grouped object and 'by' of the groupby must have the same divisions.")
        if isinstance(self.by, list):
            by_meta = [item._meta if isinstance(item, Series) else item for item in self.by]
        elif isinstance(self.by, Series):
            by_meta = self.by._meta
        else:
            by_meta = self.by
        self.dropna = {}
        if dropna is not None:
            self.dropna['dropna'] = dropna
        self.observed = {}
        if observed is not None:
            self.observed['observed'] = observed
        self._meta = self.obj._meta.groupby(by_meta, group_keys=group_keys, **self.observed, **self.dropna)

    @property
    @_deprecated()
    def index(self):
        return self.by

    @index.setter
    def index(self, value):
        self.by = value

    @property
    def _groupby_kwargs(self):
        return {'by': self.by, 'group_keys': self.group_keys, **self.dropna, 'sort': self.sort, **self.observed}

    def __iter__(self):
        raise NotImplementedError("Iteration of DataFrameGroupBy objects requires computing the groups which may be slow. You probably want to use 'apply' to execute a function for all the columns. To access individual groups, use 'get_group'. To list all the group names, use 'df[<group column>].unique().compute()'.")

    @property
    def _meta_nonempty(self):
        """
        Return a pd.DataFrameGroupBy / pd.SeriesGroupBy which contains sample data.
        """
        sample = self.obj._meta_nonempty
        if isinstance(self.by, list):
            by_meta = [item._meta_nonempty if isinstance(item, Series) else item for item in self.by]
        elif isinstance(self.by, Series):
            by_meta = self.by._meta_nonempty
        else:
            by_meta = self.by
        with check_observed_deprecation():
            grouped = sample.groupby(by_meta, group_keys=self.group_keys, **self.observed, **self.dropna)
        return _maybe_slice(grouped, self._slice)

    def _single_agg(self, token, func, aggfunc=None, meta=None, split_every=None, split_out=1, shuffle_method=None, chunk_kwargs=None, aggregate_kwargs=None, columns=None):
        """
        Aggregation with a single function/aggfunc rather than a compound spec
        like in GroupBy.aggregate
        """
        shuffle_method = _determine_split_out_shuffle(shuffle_method, split_out)
        if aggfunc is None:
            aggfunc = func
        if chunk_kwargs is None:
            chunk_kwargs = {}
        if aggregate_kwargs is None:
            aggregate_kwargs = {}
        if meta is None:
            with check_numeric_only_deprecation():
                meta = func(self._meta_nonempty, **chunk_kwargs)
        if columns is None:
            columns = meta.name if is_series_like(meta) else meta.columns
        args = [self.obj] + (self.by if isinstance(self.by, list) else [self.by])
        token = self._token_prefix + token
        levels = _determine_levels(self.by)
        if shuffle_method:
            return _shuffle_aggregate(args, chunk=_apply_chunk, chunk_kwargs={'chunk': func, 'columns': columns, **self.observed, **self.dropna, **chunk_kwargs}, aggregate=_groupby_aggregate, aggregate_kwargs={'aggfunc': aggfunc, 'levels': levels, **self.observed, **self.dropna, **aggregate_kwargs}, token=token, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, sort=self.sort)
        return aca(args, chunk=_apply_chunk, chunk_kwargs=dict(chunk=func, columns=columns, **self.observed, **chunk_kwargs, **self.dropna), aggregate=_groupby_aggregate, meta=meta, token=token, split_every=split_every, aggregate_kwargs=dict(aggfunc=aggfunc, levels=levels, **self.observed, **aggregate_kwargs, **self.dropna), split_out=split_out, split_out_setup=split_out_on_index, sort=self.sort)

    def _cum_agg(self, token, chunk, aggregate, initial, numeric_only=no_default):
        """Wrapper for cumulative groupby operation"""
        numeric_only_kwargs = get_numeric_only_kwargs(numeric_only)
        meta = chunk(self._meta, **numeric_only_kwargs)
        columns = meta.name if is_series_like(meta) else meta.columns
        by_cols = self.by if isinstance(self.by, list) else [self.by]
        if columns is not None:
            grouping_columns = [columns] if is_series_like(meta) else columns
            to_rename = set(grouping_columns) & set(by_cols)
            by = []
            for col in by_cols:
                if col in to_rename:
                    suffix = str(uuid.uuid4())
                    self.obj = self.obj.assign(**{col + suffix: self.obj[col]})
                    by.append(col + suffix)
                else:
                    by.append(col)
        else:
            by = by_cols
        name = self._token_prefix + token
        name_part = name + '-map'
        name_last = name + '-take-last'
        name_cum = name + '-cum-last'
        cumpart_raw = map_partitions(_apply_chunk, self.obj, *by, chunk=chunk, columns=columns, token=name_part, meta=meta, **self.dropna)
        cumpart_raw_frame = cumpart_raw.to_frame() if is_series_like(meta) else cumpart_raw
        cumpart_ext = cumpart_raw_frame.assign(**{i: self.obj[i] if np.isscalar(i) and i in getattr(self.obj, 'columns', []) else self.obj.index for i in by})
        grouper = grouper_dispatch(self._meta.obj)
        by_groupers = [grouper(key=ind) for ind in by]
        cumlast = map_partitions(_apply_chunk, cumpart_ext, *by_groupers, columns=0 if columns is None else columns, chunk=M.last, meta=meta, token=name_last, **self.dropna)
        _hash = tokenize(self, token, chunk, aggregate, initial)
        name += '-' + _hash
        name_cum += '-' + _hash
        dask = {}
        dask[name, 0] = (cumpart_raw._name, 0)
        for i in range(1, self.obj.npartitions):
            if i == 1:
                dask[name_cum, i] = (cumlast._name, i - 1)
            else:
                dask[name_cum, i] = (_cum_agg_filled, (name_cum, i - 1), (cumlast._name, i - 1), aggregate, initial)
            dask[name, i] = (_cum_agg_aligned, (cumpart_ext._name, i), (name_cum, i), by, 0 if columns is None else columns, aggregate, initial)
        dependencies = [cumpart_raw]
        if self.obj.npartitions > 1:
            dependencies += [cumpart_ext, cumlast]
        graph = HighLevelGraph.from_collections(name, dask, dependencies=dependencies)
        return new_dd_object(graph, name, chunk(self._meta, **numeric_only_kwargs), self.obj.divisions)

    def compute(self, **kwargs):
        raise NotImplementedError('DataFrameGroupBy does not allow compute method.Please chain it with an aggregation method (like ``.mean()``) or get a specific group using ``.get_group()`` before calling ``compute()``')

    def _shuffle(self, meta):
        df = self.obj
        if isinstance(self.obj, Series):
            df = df.to_frame('__series__')
            convert_back_to_series = True
        else:
            convert_back_to_series = False
        if isinstance(self.by, DataFrame):
            df2 = df.assign(**{'_by_' + c: self.by[c] for c in self.by.columns})
        elif isinstance(self.by, Series):
            df2 = df.assign(_by=self.by)
        else:
            df2 = df
        df3 = df2.shuffle(on=self.by)
        if isinstance(self.by, DataFrame):
            cols = ['_by_' + c for c in self.by.columns]
            by2 = df3[cols]
            if is_dataframe_like(meta):
                df4 = df3.map_partitions(drop_columns, cols, meta.columns.dtype)
            else:
                df4 = df3.drop(cols, axis=1)
        elif isinstance(self.by, Series):
            by2 = df3['_by']
            by2.name = self.by.name
            if is_dataframe_like(meta):
                df4 = df3.map_partitions(drop_columns, '_by', meta.columns.dtype)
            else:
                df4 = df3.drop('_by', axis=1)
        else:
            df4 = df3
            by2 = self.by
        if convert_back_to_series:
            df4 = df4['__series__'].rename(self.obj.name)
        return (df4, by2)

    @_deprecated_kwarg('axis')
    @derived_from(pd.core.groupby.GroupBy)
    def cumsum(self, axis=no_default, numeric_only=no_default):
        axis = self._normalize_axis(axis, 'cumsum')
        if axis:
            if axis in (1, 'columns') and isinstance(self, SeriesGroupBy):
                raise ValueError(f'No axis named {axis} for object type Series')
            return self.obj.cumsum(axis=axis)
        else:
            return self._cum_agg('cumsum', chunk=M.cumsum, aggregate=M.add, initial=0, numeric_only=numeric_only)

    @_deprecated_kwarg('axis')
    @derived_from(pd.core.groupby.GroupBy)
    def cumprod(self, axis=no_default, numeric_only=no_default):
        axis = self._normalize_axis(axis, 'cumprod')
        if axis:
            if axis in (1, 'columns') and isinstance(self, SeriesGroupBy):
                raise ValueError(f'No axis named {axis} for object type Series')
            return self.obj.cumprod(axis=axis)
        else:
            return self._cum_agg('cumprod', chunk=M.cumprod, aggregate=M.mul, initial=1, numeric_only=numeric_only)

    @_deprecated_kwarg('axis')
    @derived_from(pd.core.groupby.GroupBy)
    def cumcount(self, axis=no_default):
        return self._cum_agg('cumcount', chunk=M.cumcount, aggregate=_cumcount_aggregate, initial=-1)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.GroupBy)
    @numeric_only_deprecate_default
    def sum(self, split_every=None, split_out=1, shuffle_method=None, min_count=None, numeric_only=no_default):
        numeric_kwargs = get_numeric_only_kwargs(numeric_only)
        result = self._single_agg(func=M.sum, token='sum', split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, chunk_kwargs=numeric_kwargs, aggregate_kwargs=numeric_kwargs)
        if min_count:
            return result.where(self.count() >= min_count, other=np.nan)
        else:
            return result

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.GroupBy)
    @numeric_only_deprecate_default
    def prod(self, split_every=None, split_out=1, shuffle_method=None, min_count=None, numeric_only=no_default):
        numeric_kwargs = get_numeric_only_kwargs(numeric_only)
        result = self._single_agg(func=M.prod, token='prod', split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, chunk_kwargs=numeric_kwargs, aggregate_kwargs=numeric_kwargs)
        if min_count:
            return result.where(self.count() >= min_count, other=np.nan)
        else:
            return result

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.GroupBy)
    def min(self, split_every=None, split_out=1, shuffle_method=None, numeric_only=no_default):
        numeric_kwargs = get_numeric_only_kwargs(numeric_only)
        return self._single_agg(func=M.min, token='min', split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, chunk_kwargs=numeric_kwargs, aggregate_kwargs=numeric_kwargs)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.GroupBy)
    def max(self, split_every=None, split_out=1, shuffle_method=None, numeric_only=no_default):
        numeric_kwargs = get_numeric_only_kwargs(numeric_only)
        return self._single_agg(func=M.max, token='max', split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, chunk_kwargs=numeric_kwargs, aggregate_kwargs=numeric_kwargs)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.DataFrame)
    @numeric_only_deprecate_default
    def idxmin(self, split_every=None, split_out=1, shuffle_method=None, axis=no_default, skipna=True, numeric_only=no_default):
        if axis != no_default:
            warnings.warn('`axis` parameter is deprecated and will be removed in a future version.', FutureWarning)
        if axis in (1, 'columns'):
            raise NotImplementedError(f'The axis={axis} keyword is not implemented for groupby.idxmin')
        self._normalize_axis(axis, 'idxmin')
        chunk_kwargs = dict(skipna=skipna)
        numeric_kwargs = get_numeric_only_kwargs(numeric_only)
        chunk_kwargs.update(numeric_kwargs)
        return self._single_agg(func=M.idxmin, token='idxmin', aggfunc=M.first, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, chunk_kwargs=chunk_kwargs, aggregate_kwargs=numeric_kwargs)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.DataFrame)
    @numeric_only_deprecate_default
    def idxmax(self, split_every=None, split_out=1, shuffle_method=None, axis=no_default, skipna=True, numeric_only=no_default):
        if axis != no_default:
            warnings.warn('`axis` parameter is deprecated and will be removed in a future version.', FutureWarning)
        if axis in (1, 'columns'):
            raise NotImplementedError(f'The axis={axis} keyword is not implemented for groupby.idxmax')
        self._normalize_axis(axis, 'idxmax')
        chunk_kwargs = dict(skipna=skipna)
        numeric_kwargs = get_numeric_only_kwargs(numeric_only)
        chunk_kwargs.update(numeric_kwargs)
        return self._single_agg(func=M.idxmax, token='idxmax', aggfunc=M.first, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, chunk_kwargs=chunk_kwargs, aggregate_kwargs=numeric_kwargs)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.GroupBy)
    def count(self, split_every=None, split_out=1, shuffle_method=None):
        return self._single_agg(func=M.count, token='count', aggfunc=M.sum, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.GroupBy)
    @numeric_only_not_implemented
    def mean(self, split_every=None, split_out=1, shuffle_method=None, numeric_only=no_default):
        with check_numeric_only_deprecation():
            s = self.sum(split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, numeric_only=numeric_only)
        c = self.count(split_every=split_every, split_out=split_out, shuffle_method=shuffle_method)
        if is_dataframe_like(s):
            c = c[s.columns]
        return s / c

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.GroupBy)
    def median(self, split_every=None, split_out=1, shuffle_method=None, numeric_only=no_default):
        if shuffle_method is False:
            raise ValueError("In order to aggregate with 'median', you must use shuffling-based aggregation (e.g., shuffle='tasks')")
        shuffle_method = shuffle_method or _determine_split_out_shuffle(True, split_out)
        numeric_only_kwargs = get_numeric_only_kwargs(numeric_only)
        with check_numeric_only_deprecation(name='median'):
            meta = self._meta_nonempty.median(**numeric_only_kwargs)
        columns = meta.name if is_series_like(meta) else meta.columns
        by = self.by if isinstance(self.by, list) else [self.by]
        return _shuffle_aggregate([self.obj] + by, token='non-agg', chunk=_non_agg_chunk, chunk_kwargs={'key': columns, **self.observed, **self.dropna}, aggregate=_groupby_aggregate, aggregate_kwargs={'aggfunc': _median_aggregate, 'levels': _determine_levels(self.by), **self.observed, **self.dropna, **numeric_only_kwargs}, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, sort=self.sort)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.GroupBy)
    def size(self, split_every=None, split_out=1, shuffle_method=None):
        return self._single_agg(token='size', func=M.size, aggfunc=M.sum, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method)

    @derived_from(pd.core.groupby.GroupBy)
    @numeric_only_not_implemented
    def var(self, ddof=1, split_every=None, split_out=1, numeric_only=no_default):
        if not PANDAS_GE_150 and numeric_only is not no_default:
            raise TypeError('numeric_only not supported for pandas < 1.5')
        levels = _determine_levels(self.by)
        result = aca([self.obj, self.by] if not isinstance(self.by, list) else [self.obj] + self.by, chunk=_var_chunk, aggregate=_var_agg, combine=_var_combine, token=self._token_prefix + 'var', aggregate_kwargs={'ddof': ddof, 'levels': levels, 'numeric_only': numeric_only, **self.observed, **self.dropna}, chunk_kwargs={'numeric_only': numeric_only, **self.observed, **self.dropna}, combine_kwargs={'levels': levels}, split_every=split_every, split_out=split_out, split_out_setup=split_out_on_index, sort=self.sort)
        if isinstance(self.obj, Series):
            result = result[result.columns[0]]
        if self._slice:
            result = result[self._slice]
        return result

    @derived_from(pd.core.groupby.GroupBy)
    @numeric_only_not_implemented
    def std(self, ddof=1, split_every=None, split_out=1, numeric_only=no_default):
        if not PANDAS_GE_150 and numeric_only is not no_default:
            raise TypeError('numeric_only not supported for pandas < 1.5')
        with check_numeric_only_deprecation():
            v = self.var(ddof, split_every=split_every, split_out=split_out, numeric_only=numeric_only)
        result = map_partitions(np.sqrt, v, meta=v)
        return result

    @derived_from(pd.DataFrame)
    def corr(self, ddof=1, split_every=None, split_out=1, numeric_only=no_default):
        """Groupby correlation:
        corr(X, Y) = cov(X, Y) / (std_x * std_y)
        """
        if not PANDAS_GE_150 and numeric_only is not no_default:
            raise TypeError('numeric_only not supported for pandas < 1.5')
        return self.cov(split_every=split_every, split_out=split_out, std=True, numeric_only=numeric_only)

    @derived_from(pd.DataFrame)
    def cov(self, ddof=1, split_every=None, split_out=1, std=False, numeric_only=no_default):
        """Groupby covariance is accomplished by

        1. Computing intermediate values for sum, count, and the product of
           all columns: a b c -> a*a, a*b, b*b, b*c, c*c.

        2. The values are then aggregated and the final covariance value is calculated:
           cov(X, Y) = X*Y - Xbar * Ybar

        When `std` is True calculate Correlation
        """
        if not PANDAS_GE_150 and numeric_only is not no_default:
            raise TypeError('numeric_only not supported for pandas < 1.5')
        numeric_only_kwargs = get_numeric_only_kwargs(numeric_only)
        levels = _determine_levels(self.by)
        is_mask = any((is_series_like(s) for s in self.by))
        if self._slice:
            if is_mask:
                self.obj = self.obj[self._slice]
            else:
                sliced_plus = list(self._slice) + list(self.by)
                self.obj = self.obj[sliced_plus]
        result = aca([self.obj, self.by] if not isinstance(self.by, list) else [self.obj] + self.by, chunk=_cov_chunk, aggregate=_cov_agg, combine=_cov_combine, token=self._token_prefix + 'cov', aggregate_kwargs={'ddof': ddof, 'levels': levels, 'std': std}, combine_kwargs={'levels': levels}, chunk_kwargs=numeric_only_kwargs, split_every=split_every, split_out=split_out, split_out_setup=split_out_on_index, sort=self.sort)
        if isinstance(self.obj, Series):
            result = result[result.columns[0]]
        if self._slice:
            result = result[self._slice]
        return result

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.GroupBy)
    def first(self, split_every=None, split_out=1, shuffle_method=None, numeric_only=no_default):
        numeric_kwargs = get_numeric_only_kwargs(numeric_only)
        return self._single_agg(func=M.first, token='first', split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, chunk_kwargs=numeric_kwargs, aggregate_kwargs=numeric_kwargs)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.GroupBy)
    def last(self, split_every=None, split_out=1, shuffle_method=None, numeric_only=no_default):
        numeric_kwargs = get_numeric_only_kwargs(numeric_only)
        return self._single_agg(token='last', func=M.last, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, chunk_kwargs=numeric_kwargs, aggregate_kwargs=numeric_kwargs)

    @derived_from(pd.core.groupby.GroupBy, inconsistencies='If the group is not present, Dask will return an empty Series/DataFrame.')
    def get_group(self, key):
        token = self._token_prefix + 'get_group'
        meta = self._meta.obj
        if is_dataframe_like(meta) and self._slice is not None:
            meta = meta[self._slice]
        columns = meta.columns if is_dataframe_like(meta) else meta.name
        return map_partitions(_groupby_get_group, self.obj, self.by, key, columns, meta=meta, token=token)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @_aggregate_docstring()
    def aggregate(self, arg=None, split_every=None, split_out=1, shuffle_method=None, **kwargs):
        if split_out is None:
            warnings.warn('split_out=None is deprecated, please use a positive integer, or allow the default of 1', category=FutureWarning)
            split_out = 1
        shuffle_method = _determine_split_out_shuffle(shuffle_method, split_out)
        relabeling = None
        columns = None
        order = None
        column_projection = None
        if PANDAS_GE_140:
            if isinstance(self, DataFrameGroupBy):
                if arg is None:
                    relabeling, arg, columns, order = reconstruct_func(arg, **kwargs)
            elif isinstance(self, SeriesGroupBy):
                relabeling = arg is None
                if relabeling:
                    columns, arg = validate_func_kwargs(kwargs)
        if isinstance(self.obj, DataFrame):
            if isinstance(self.by, tuple) or np.isscalar(self.by):
                group_columns = {self.by}
            elif isinstance(self.by, list):
                group_columns = {i for i in self.by if isinstance(i, tuple) or np.isscalar(i)}
            else:
                group_columns = set()
            if self._slice:
                non_group_columns = self._slice
                if not isinstance(non_group_columns, list):
                    non_group_columns = [non_group_columns]
            else:
                non_group_columns = [col for col in self.obj.columns if col not in group_columns]
            spec = _normalize_spec(arg, non_group_columns)
            if isinstance(arg, dict):
                column_projection = group_columns.union(arg.keys()).intersection(self.obj.columns)
        elif isinstance(self.obj, Series):
            if isinstance(arg, (list, tuple, dict)):
                spec = _normalize_spec({None: arg}, [])
                spec = [(result_column, func, input_column) for (_, result_column), func, input_column in spec]
            else:
                spec = _normalize_spec({None: arg}, [])
                spec = [(self.obj.name, func, input_column) for _, func, input_column in spec]
        else:
            raise ValueError(f'aggregate on unknown object {self.obj}')
        chunk_funcs, aggregate_funcs, finalizers = _build_agg_args(spec)
        if isinstance(self.by, (tuple, list)) and len(self.by) > 1:
            levels = list(range(len(self.by)))
        else:
            levels = 0
        _obj = self.obj[list(column_projection)] if column_projection else self.obj
        if not isinstance(self.by, list):
            chunk_args = [_obj, self.by]
        else:
            chunk_args = [_obj] + self.by
        has_median = any((s[1] in ('median', np.median) for s in spec))
        if has_median and (not shuffle_method):
            raise ValueError("In order to aggregate with 'median', you must use shuffling-based aggregation (e.g., shuffle='tasks')")
        if shuffle_method:
            if has_median:
                result = _shuffle_aggregate(chunk_args, chunk=_non_agg_chunk, chunk_kwargs={'key': [c for c in _obj.columns.tolist() if c not in group_columns], **self.observed, **self.dropna}, aggregate=_groupby_aggregate_spec, aggregate_kwargs={'spec': arg, 'levels': _determine_levels(self.by), **self.observed, **self.dropna}, token='aggregate', split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, sort=self.sort)
            else:
                result = _shuffle_aggregate(chunk_args, chunk=_groupby_apply_funcs, chunk_kwargs={'funcs': chunk_funcs, 'sort': self.sort, **self.observed, **self.dropna}, aggregate=_agg_finalize, aggregate_kwargs=dict(aggregate_funcs=aggregate_funcs, finalize_funcs=finalizers, level=levels, **self.observed, **self.dropna), token='aggregate', split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, sort=self.sort)
        else:
            if self.sort and split_out > 1:
                raise NotImplementedError('Cannot guarantee sorted keys for `split_out>1` and `shuffle=False` Try using `shuffle=True` if you are grouping on a single column. Otherwise, try using split_out=1, or grouping with sort=False.')
            result = aca(chunk_args, chunk=_groupby_apply_funcs, chunk_kwargs=dict(funcs=chunk_funcs, sort=False, **self.observed, **self.dropna), combine=_groupby_apply_funcs, combine_kwargs=dict(funcs=aggregate_funcs, level=levels, sort=False, **self.observed, **self.dropna), aggregate=_agg_finalize, aggregate_kwargs=dict(aggregate_funcs=aggregate_funcs, finalize_funcs=finalizers, level=levels, **self.observed, **self.dropna), token='aggregate', split_every=split_every, split_out=split_out, split_out_setup=split_out_on_index, sort=self.sort)
        if relabeling and result is not None:
            if order is not None:
                result = result.iloc[:, order]
            result.columns = columns
        return result

    @insert_meta_param_description(pad=12)
    def apply(self, func, *args, **kwargs):
        """Parallel version of pandas GroupBy.apply

        This mimics the pandas version except for the following:

        1.  If the grouper does not align with the index then this causes a full
            shuffle.  The order of rows within each group may not be preserved.
        2.  Dask's GroupBy.apply is not appropriate for aggregations. For custom
            aggregations, use :class:`dask.dataframe.groupby.Aggregation`.

        .. warning::

           Pandas' groupby-apply can be used to to apply arbitrary functions,
           including aggregations that result in one row per group. Dask's
           groupby-apply will apply ``func`` once on each group, doing a shuffle
           if needed, such that each group is contained in one partition.
           When ``func`` is a reduction, e.g., you'll end up with one row
           per group. To apply a custom aggregation with Dask,
           use :class:`dask.dataframe.groupby.Aggregation`.

        Parameters
        ----------
        func: function
            Function to apply
        args, kwargs : Scalar, Delayed or object
            Arguments and keywords to pass to the function.
        $META

        Returns
        -------
        applied : Series or DataFrame depending on columns keyword
        """
        meta = kwargs.get('meta', no_default)
        if meta is no_default:
            with raise_on_meta_error(f'groupby.apply({funcname(func)})', udf=True):
                meta_args, meta_kwargs = _extract_meta((args, kwargs), nonempty=True)
                meta = self._meta_nonempty.apply(func, *meta_args, **meta_kwargs)
            msg = "`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.\n  Before: .apply(func)\n  After:  .apply(func, meta={'x': 'f8', 'y': 'f8'}) for dataframe result\n  or:     .apply(func, meta=('x', 'f8'))            for series result"
            warnings.warn(msg, stacklevel=2)
        meta = make_meta(meta, parent_meta=self._meta.obj)
        if isinstance(self.by, list) and any((isinstance(item, Series) for item in self.by)):
            raise NotImplementedError('groupby-apply with a multiple Series is currently not supported')
        df = self.obj
        should_shuffle = not (df.known_divisions and df._contains_index_name(self.by))
        if should_shuffle:
            df2, by = self._shuffle(meta)
        else:
            df2 = df
            by = self.by
        kwargs['meta'] = meta
        df3 = map_partitions(_groupby_slice_apply, df2, by, self._slice, func, *args, token=funcname(func), group_keys=self.group_keys, **self.observed, **self.dropna, **kwargs)
        return df3

    @insert_meta_param_description(pad=12)
    def transform(self, func, *args, **kwargs):
        """Parallel version of pandas GroupBy.transform

        This mimics the pandas version except for the following:

        1.  If the grouper does not align with the index then this causes a full
            shuffle.  The order of rows within each group may not be preserved.
        2.  Dask's GroupBy.transform is not appropriate for aggregations. For custom
            aggregations, use :class:`dask.dataframe.groupby.Aggregation`.

        .. warning::

           Pandas' groupby-transform can be used to apply arbitrary functions,
           including aggregations that result in one row per group. Dask's
           groupby-transform will apply ``func`` once on each group, doing a shuffle
           if needed, such that each group is contained in one partition.
           When ``func`` is a reduction, e.g., you'll end up with one row
           per group. To apply a custom aggregation with Dask,
           use :class:`dask.dataframe.groupby.Aggregation`.

        Parameters
        ----------
        func: function
            Function to apply
        args, kwargs : Scalar, Delayed or object
            Arguments and keywords to pass to the function.
        $META

        Returns
        -------
        applied : Series or DataFrame depending on columns keyword
        """
        meta = kwargs.get('meta', no_default)
        if meta is no_default:
            with raise_on_meta_error(f'groupby.transform({funcname(func)})', udf=True):
                meta_args, meta_kwargs = _extract_meta((args, kwargs), nonempty=True)
                meta = self._meta_nonempty.transform(func, *meta_args, **meta_kwargs)
            msg = "`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.\n  Before: .transform(func)\n  After:  .transform(func, meta={'x': 'f8', 'y': 'f8'}) for dataframe result\n  or:     .transform(func, meta=('x', 'f8'))            for series result"
            warnings.warn(msg, stacklevel=2)
        meta = make_meta(meta, parent_meta=self._meta.obj)
        if isinstance(self.by, list) and any((isinstance(item, Series) for item in self.by)):
            raise NotImplementedError('groupby-transform with a multiple Series is currently not supported')
        df = self.obj
        should_shuffle = not (df.known_divisions and df._contains_index_name(self.by))
        if should_shuffle:
            df2, by = self._shuffle(meta)
        else:
            df2 = df
            by = self.by
        kwargs['meta'] = meta
        df3 = map_partitions(_groupby_slice_transform, df2, by, self._slice, func, *args, token=funcname(func), group_keys=self.group_keys, **self.observed, **self.dropna, **kwargs)
        if isinstance(self, DataFrameGroupBy):
            index_name = df3.index.name
            df3 = df3.reset_index().set_index(index_name or 'index')
            df3.index = df3.index.rename(index_name)
        return df3

    @insert_meta_param_description(pad=12)
    def shift(self, periods=1, freq=no_default, axis=no_default, fill_value=no_default, meta=no_default):
        """Parallel version of pandas GroupBy.shift

        This mimics the pandas version except for the following:

        If the grouper does not align with the index then this causes a full
        shuffle.  The order of rows within each group may not be preserved.

        Parameters
        ----------
        periods : Delayed, Scalar or int, default 1
            Number of periods to shift.
        freq : Delayed, Scalar or str, optional
            Frequency string.
        axis : axis to shift, default 0
            Shift direction.
        fill_value : Scalar, Delayed or object, optional
            The scalar value to use for newly introduced missing values.
        $META

        Returns
        -------
        shifted : Series or DataFrame shifted within each group.

        Examples
        --------
        >>> import dask
        >>> ddf = dask.datasets.timeseries(freq="1h")
        >>> result = ddf.groupby("name").shift(1, meta={"id": int, "x": float, "y": float})
        """
        if axis != no_default:
            warnings.warn('`axis` parameter is deprecated and will be removed in a future version.', FutureWarning)
        axis = self._normalize_axis(axis, 'shift')
        kwargs = {'periods': periods, 'axis': axis}
        if freq is not no_default:
            kwargs.update({'freq': freq})
        if fill_value is not no_default:
            kwargs.update({'fill_value': fill_value})
        if meta is no_default:
            with raise_on_meta_error('groupby.shift()', udf=False):
                meta_kwargs = _extract_meta(kwargs, nonempty=True)
                with check_groupby_axis_deprecation():
                    meta = self._meta_nonempty.shift(**meta_kwargs)
            msg = "`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.\n  Before: .shift(1)\n  After:  .shift(1, meta={'x': 'f8', 'y': 'f8'}) for dataframe result\n  or:     .shift(1, meta=('x', 'f8'))            for series result"
            warnings.warn(msg, stacklevel=2)
        meta = make_meta(meta, parent_meta=self._meta.obj)
        if isinstance(self.by, list) and any((isinstance(item, Series) for item in self.by)):
            raise NotImplementedError('groupby-shift with a multiple Series is currently not supported')
        df = self.obj
        should_shuffle = not (df.known_divisions and df._contains_index_name(self.by))
        if should_shuffle:
            df2, by = self._shuffle(meta)
        else:
            df2 = df
            by = self.by
        result = map_partitions(_groupby_slice_shift, df2, by, self._slice, should_shuffle, token='groupby-shift', group_keys=self.group_keys, meta=meta, **self.observed, **self.dropna, **kwargs)
        return result

    def rolling(self, window, min_periods=None, center=False, win_type=None, axis=0):
        """Provides rolling transformations.

        .. note::

            Since MultiIndexes are not well supported in Dask, this method returns a
            dataframe with the same index as the original data. The groupby column is
            not added as the first level of the index like pandas does.

            This method works differently from other groupby methods. It does a groupby
            on each partition (plus some overlap). This means that the output has the
            same shape and number of partitions as the original.

        Parameters
        ----------
        window : str, offset
           Size of the moving window. This is the number of observations used
           for calculating the statistic. Data must have a ``DatetimeIndex``
        min_periods : int, default None
            Minimum number of observations in window required to have a value
            (otherwise result is NA).
        center : boolean, default False
            Set the labels at the center of the window.
        win_type : string, default None
            Provide a window type. The recognized window types are identical
            to pandas.
        axis : int, default 0

        Returns
        -------
        a Rolling object on which to call a method to compute a statistic

        Examples
        --------
        >>> import dask
        >>> ddf = dask.datasets.timeseries(freq="1h")
        >>> result = ddf.groupby("name").x.rolling('1D').max()
        """
        from dask.dataframe.rolling import RollingGroupby
        if isinstance(window, Integral):
            raise ValueError("Only time indexes are supported for rolling groupbys in dask dataframe. ``window`` must be a ``freq`` (e.g. '1H').")
        if min_periods is not None:
            if not isinstance(min_periods, Integral):
                raise ValueError('min_periods must be an integer')
            if min_periods < 0:
                raise ValueError('min_periods must be >= 0')
        return RollingGroupby(self, window=window, min_periods=min_periods, center=center, win_type=win_type, axis=axis)

    def _normalize_axis(self, axis, method: str):
        if PANDAS_GE_210 and axis is not no_default:
            if axis in (0, 'index'):
                warnings.warn(f"The 'axis' keyword in {type(self).__name__}.{method} is deprecated and will be removed in a future version. Call without passing 'axis' instead.", FutureWarning)
            else:
                warnings.warn(f'{type(self).__name__}.{method} with axis={axis} is deprecated and will be removed in a future version. Operate on the un-grouped DataFrame instead', FutureWarning)
        if axis is no_default:
            axis = 0
        if axis in ('index', 1):
            warnings.warn('Using axis=1 in GroupBy does not require grouping and will be removed entirely in a future version.', FutureWarning)
        return axis

    @_deprecated(message='Please use `ffill`/`bfill` or `fillna` without a GroupBy.')
    def fillna(self, value=None, method=None, limit=None, axis=no_default):
        """Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, default None
            Value to use to fill holes (e.g. 0).
        method : {'bfill', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series. ffill: propagate last
            valid observation forward to next valid. bfill: use next valid observation
            to fill gap.
        axis : {0 or 'index', 1 or 'columns'}
            Axis along which to fill missing values.
        limit : int, default None
            If method is specified, this is the maximum number of consecutive NaN values
            to forward/backward fill. In other words, if there is a gap with more than
            this number of consecutive NaNs, it will only be partially filled. If method
            is not specified, this is the maximum number of entries along the entire
            axis where NaNs will be filled. Must be greater than 0 if not None.

        Returns
        -------
        Series or DataFrame
            Object with missing values filled

        See also
        --------
        pandas.core.groupby.DataFrameGroupBy.fillna
        """
        axis = self._normalize_axis(axis, 'fillna')
        if not np.isscalar(value) and value is not None:
            raise NotImplementedError('groupby-fillna with value=dict/Series/DataFrame is not supported')
        kwargs = dict(value=value, method=method, limit=limit, axis=axis)
        if PANDAS_GE_220:
            func = M.fillna
            kwargs.update(include_groups=False)
        else:
            func = _drop_apply
            kwargs.update(by=self.by, what='fillna')
        meta = self._meta_nonempty.apply(func, **kwargs)
        result = self.apply(func, meta=meta, **kwargs)
        if PANDAS_GE_150 and self.group_keys:
            return result.map_partitions(M.droplevel, self.by)
        return result

    @derived_from(pd.core.groupby.GroupBy)
    def ffill(self, limit=None):
        kwargs = dict(limit=limit)
        if PANDAS_GE_220:
            func = M.ffill
            kwargs.update(include_groups=False)
        else:
            func = _drop_apply
            kwargs.update(by=self.by, what='ffill')
        meta = self._meta_nonempty.apply(func, **kwargs)
        result = self.apply(func, meta=meta, **kwargs)
        if PANDAS_GE_150 and self.group_keys:
            return result.map_partitions(M.droplevel, self.by)
        return result

    @derived_from(pd.core.groupby.GroupBy)
    def bfill(self, limit=None):
        kwargs = dict(limit=limit)
        if PANDAS_GE_220:
            func = M.bfill
            kwargs.update(include_groups=False)
        else:
            func = _drop_apply
            kwargs.update(by=self.by, what='bfill')
        meta = self._meta_nonempty.apply(func, **kwargs)
        result = self.apply(func, meta=meta, **kwargs)
        if PANDAS_GE_150 and self.group_keys:
            return result.map_partitions(M.droplevel, self.by)
        return result