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