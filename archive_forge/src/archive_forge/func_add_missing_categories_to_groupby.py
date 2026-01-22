import abc
from collections import namedtuple
from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._libs.tslibs import to_offset
from pandas.core.dtypes.common import is_list_like, is_numeric_dtype
from pandas.core.resample import _get_timestamp_range_edges
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
def add_missing_categories_to_groupby(dfs, by, operator, initial_columns, combined_cols, is_udf_agg, kwargs, initial_dtypes=None):
    """
    Generate values for missing categorical values to be inserted into groupby result.

    This function is used to emulate behavior of ``groupby(observed=False)`` parameter,
    it takes groupby result that was computed using ``groupby(observed=True)``
    and computes results for categorical values that are not presented in `dfs`.

    Parameters
    ----------
    dfs : list of pandas.DataFrames
        Row partitions containing groupby results.
    by : list of hashable
        Column labels that were used to perform groupby.
    operator : callable
        Aggregation function that was used during groupby.
    initial_columns : pandas.Index
        Column labels of the original dataframe.
    combined_cols : pandas.Index
        Column labels of the groupby result.
    is_udf_agg : bool
        Whether ``operator`` is a UDF.
    kwargs : dict
        Parameters that were passed to ``groupby(by, **kwargs)``.
    initial_dtypes : pandas.Series, optional
        Dtypes of the original dataframe. If not specified, assume it's ``int64``.

    Returns
    -------
    masks : dict[int, pandas.DataFrame]
        Mapping between partition idx and a dataframe with results for missing categorical values
        to insert to this partition.
    new_combined_cols : pandas.Index
        New column labels of the groupby result. If ``is_udf_agg is True``, then ``operator``
        may change the resulted columns.
    """
    kwargs['observed'] = False
    new_combined_cols = combined_cols
    indices = [df.index for df in dfs]
    total_index = indices[0].append(indices[1:])
    if isinstance(total_index, pandas.MultiIndex):
        if all((not isinstance(level, pandas.CategoricalIndex) for level in total_index.levels)):
            return ({}, new_combined_cols)
        missing_cats_dtype = {name: level.dtype if isinstance(level.dtype, pandas.CategoricalDtype) else pandas.CategoricalDtype(level) for level, name in zip(total_index.levels, total_index.names)}
        complete_index = pandas.MultiIndex.from_product([value.categories.astype(total_level.dtype) for total_level, value in zip(total_index.levels, missing_cats_dtype.values())], names=by)
        missing_index = complete_index[~complete_index.isin(total_index)]
    else:
        if not isinstance(total_index, pandas.CategoricalIndex):
            return ({}, new_combined_cols)
        missing_index = total_index.categories.difference(total_index.values)
        missing_cats_dtype = {by[0]: pandas.CategoricalDtype(missing_index)}
    missing_index.names = by
    if len(missing_index) == 0:
        return ({}, new_combined_cols)
    if is_udf_agg and isinstance(total_index, pandas.MultiIndex):
        missing_values = pandas.DataFrame({0: [np.NaN]})
    else:
        if not is_udf_agg:
            missing_cats_dtype = {key: pandas.CategoricalDtype(value.categories[:1]) for key, value in missing_cats_dtype.items()}
        empty_df = pandas.DataFrame(columns=initial_columns)
        empty_df = empty_df.astype('int64' if initial_dtypes is None else initial_dtypes)
        empty_df = empty_df.astype(missing_cats_dtype)
        missing_values = operator(empty_df.groupby(by, **kwargs))
    if is_udf_agg and (not isinstance(total_index, pandas.MultiIndex)):
        missing_values = missing_values.drop(columns=by, errors='ignore')
        new_combined_cols = pandas.concat([pandas.DataFrame(columns=combined_cols), missing_values.iloc[:0]], axis=0, join='outer').columns
    else:
        fill_value = np.NaN if len(missing_values) == 0 else missing_values.iloc[0, 0]
        missing_values = pandas.DataFrame(fill_value, index=missing_index, columns=combined_cols)
    if not isinstance(missing_values.index, pandas.MultiIndex):
        missing_values.index = missing_values.index.astype(total_index.dtype)
    if not kwargs['sort']:
        mask = {len(indices) - 1: missing_values}
        return (mask, new_combined_cols)
    bins = []
    old_bins_to_new = {}
    offset = 0
    for idx in indices[:-1]:
        if len(idx) == 0:
            offset += 1
            continue
        old_bins_to_new[len(bins)] = offset
        bins.append(idx[-1][0] if isinstance(idx, pandas.MultiIndex) else idx[-1])
    old_bins_to_new[len(bins)] = offset
    if len(bins) == 0:
        return ({old_bins_to_new.get(0, 0): missing_values}, new_combined_cols)
    lvl_zero = missing_values.index.levels[0] if isinstance(missing_values.index, pandas.MultiIndex) else missing_values.index
    if pandas.api.types.is_any_real_numeric_dtype(lvl_zero):
        part_idx = np.digitize(lvl_zero, bins, right=True)
    else:
        part_idx = np.searchsorted(bins, lvl_zero)
    masks = {}
    if isinstance(total_index, pandas.MultiIndex):
        for idx, values in pandas.RangeIndex(len(lvl_zero)).groupby(part_idx).items():
            masks[idx] = missing_values[pandas.Index(missing_values.index.codes[0]).isin(values)]
    else:
        frame_idx = missing_values.index.to_frame()
        for idx, values in lvl_zero.groupby(part_idx).items():
            masks[idx] = missing_values[frame_idx.iloc[:, 0].isin(values)]
    masks = {key + old_bins_to_new[key]: value for key, value in masks.items()}
    return (masks, new_combined_cols)