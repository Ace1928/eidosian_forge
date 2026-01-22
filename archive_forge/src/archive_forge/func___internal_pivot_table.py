from __future__ import annotations
from collections.abc import (
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
import pandas.core.common as com
from pandas.core.frame import _shared_docs
from pandas.core.groupby import Grouper
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.reshape.util import cartesian_product
from pandas.core.series import Series
def __internal_pivot_table(data: DataFrame, values, index, columns, aggfunc: AggFuncTypeBase | AggFuncTypeDict, fill_value, margins: bool, dropna: bool, margins_name: Hashable, observed: bool | lib.NoDefault, sort: bool) -> DataFrame:
    """
    Helper of :func:`pandas.pivot_table` for any non-list ``aggfunc``.
    """
    keys = index + columns
    values_passed = values is not None
    if values_passed:
        if is_list_like(values):
            values_multi = True
            values = list(values)
        else:
            values_multi = False
            values = [values]
        for i in values:
            if i not in data:
                raise KeyError(i)
        to_filter = []
        for x in keys + values:
            if isinstance(x, Grouper):
                x = x.key
            try:
                if x in data:
                    to_filter.append(x)
            except TypeError:
                pass
        if len(to_filter) < len(data.columns):
            data = data[to_filter]
    else:
        values = data.columns
        for key in keys:
            try:
                values = values.drop(key)
            except (TypeError, ValueError, KeyError):
                pass
        values = list(values)
    observed_bool = False if observed is lib.no_default else observed
    grouped = data.groupby(keys, observed=observed_bool, sort=sort, dropna=dropna)
    if observed is lib.no_default and any((ping._passed_categorical for ping in grouped._grouper.groupings)):
        warnings.warn('The default value of observed=False is deprecated and will change to observed=True in a future version of pandas. Specify observed=False to silence this warning and retain the current behavior', category=FutureWarning, stacklevel=find_stack_level())
    agged = grouped.agg(aggfunc)
    if dropna and isinstance(agged, ABCDataFrame) and len(agged.columns):
        agged = agged.dropna(how='all')
    table = agged
    if table.index.nlevels > 1 and index:
        index_names = agged.index.names[:len(index)]
        to_unstack = []
        for i in range(len(index), len(keys)):
            name = agged.index.names[i]
            if name is None or name in index_names:
                to_unstack.append(i)
            else:
                to_unstack.append(name)
        table = agged.unstack(to_unstack, fill_value=fill_value)
    if not dropna:
        if isinstance(table.index, MultiIndex):
            m = MultiIndex.from_arrays(cartesian_product(table.index.levels), names=table.index.names)
            table = table.reindex(m, axis=0, fill_value=fill_value)
        if isinstance(table.columns, MultiIndex):
            m = MultiIndex.from_arrays(cartesian_product(table.columns.levels), names=table.columns.names)
            table = table.reindex(m, axis=1, fill_value=fill_value)
    if sort is True and isinstance(table, ABCDataFrame):
        table = table.sort_index(axis=1)
    if fill_value is not None:
        table = table.fillna(fill_value)
        if aggfunc is len and (not observed) and lib.is_integer(fill_value):
            table = table.astype(np.int64)
    if margins:
        if dropna:
            data = data[data.notna().all(axis=1)]
        table = _add_margins(table, data, values, rows=index, cols=columns, aggfunc=aggfunc, observed=dropna, margins_name=margins_name, fill_value=fill_value)
    if values_passed and (not values_multi) and (table.columns.nlevels > 1):
        table.columns = table.columns.droplevel(0)
    if len(index) == 0 and len(columns) > 0:
        table = table.T
    if isinstance(table, ABCDataFrame) and dropna:
        table = table.dropna(how='all', axis=1)
    return table