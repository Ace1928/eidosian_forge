from __future__ import annotations
import textwrap
from typing import (
import numpy as np
from pandas._libs import (
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.cast import find_common_type
from pandas.core.algorithms import safe_sort
from pandas.core.indexes.base import (
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
def union_indexes(indexes, sort: bool | None=True) -> Index:
    """
    Return the union of indexes.

    The behavior of sort and names is not consistent.

    Parameters
    ----------
    indexes : list of Index or list objects
    sort : bool, default True
        Whether the result index should come out sorted or not.

    Returns
    -------
    Index
    """
    if len(indexes) == 0:
        raise AssertionError('Must have at least 1 Index to union')
    if len(indexes) == 1:
        result = indexes[0]
        if isinstance(result, list):
            if not sort:
                result = Index(result)
            else:
                result = Index(sorted(result))
        return result
    indexes, kind = _sanitize_and_check(indexes)

    def _unique_indices(inds, dtype) -> Index:
        """
        Concatenate indices and remove duplicates.

        Parameters
        ----------
        inds : list of Index or list objects
        dtype : dtype to set for the resulting Index

        Returns
        -------
        Index
        """
        if all((isinstance(ind, Index) for ind in inds)):
            inds = [ind.astype(dtype, copy=False) for ind in inds]
            result = inds[0].unique()
            other = inds[1].append(inds[2:])
            diff = other[result.get_indexer_for(other) == -1]
            if len(diff):
                result = result.append(diff.unique())
            if sort:
                result = result.sort_values()
            return result

        def conv(i):
            if isinstance(i, Index):
                i = i.tolist()
            return i
        return Index(lib.fast_unique_multiple_list([conv(i) for i in inds], sort=sort), dtype=dtype)

    def _find_common_index_dtype(inds):
        """
        Finds a common type for the indexes to pass through to resulting index.

        Parameters
        ----------
        inds: list of Index or list objects

        Returns
        -------
        The common type or None if no indexes were given
        """
        dtypes = [idx.dtype for idx in indexes if isinstance(idx, Index)]
        if dtypes:
            dtype = find_common_type(dtypes)
        else:
            dtype = None
        return dtype
    if kind == 'special':
        result = indexes[0]
        dtis = [x for x in indexes if isinstance(x, DatetimeIndex)]
        dti_tzs = [x for x in dtis if x.tz is not None]
        if len(dti_tzs) not in [0, len(dtis)]:
            raise TypeError('Cannot join tz-naive with tz-aware DatetimeIndex')
        if len(dtis) == len(indexes):
            sort = True
            result = indexes[0]
        elif len(dtis) > 1:
            sort = False
            indexes = [x.astype(object, copy=False) for x in indexes]
            result = indexes[0]
        for other in indexes[1:]:
            result = result.union(other, sort=None if sort else False)
        return result
    elif kind == 'array':
        dtype = _find_common_index_dtype(indexes)
        index = indexes[0]
        if not all((index.equals(other) for other in indexes[1:])):
            index = _unique_indices(indexes, dtype)
        name = get_unanimous_names(*indexes)[0]
        if name != index.name:
            index = index.rename(name)
        return index
    else:
        dtype = _find_common_index_dtype(indexes)
        return _unique_indices(indexes, dtype)