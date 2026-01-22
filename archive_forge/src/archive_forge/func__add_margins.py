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
def _add_margins(table: DataFrame | Series, data: DataFrame, values, rows, cols, aggfunc, observed: bool, margins_name: Hashable='All', fill_value=None):
    if not isinstance(margins_name, str):
        raise ValueError('margins_name argument must be a string')
    msg = f'Conflicting name "{margins_name}" in margins'
    for level in table.index.names:
        if margins_name in table.index.get_level_values(level):
            raise ValueError(msg)
    grand_margin = _compute_grand_margin(data, values, aggfunc, margins_name)
    if table.ndim == 2:
        for level in table.columns.names[1:]:
            if margins_name in table.columns.get_level_values(level):
                raise ValueError(msg)
    key: str | tuple[str, ...]
    if len(rows) > 1:
        key = (margins_name,) + ('',) * (len(rows) - 1)
    else:
        key = margins_name
    if not values and isinstance(table, ABCSeries):
        return table._append(table._constructor({key: grand_margin[margins_name]}))
    elif values:
        marginal_result_set = _generate_marginal_results(table, data, values, rows, cols, aggfunc, observed, margins_name)
        if not isinstance(marginal_result_set, tuple):
            return marginal_result_set
        result, margin_keys, row_margin = marginal_result_set
    else:
        assert isinstance(table, ABCDataFrame)
        marginal_result_set = _generate_marginal_results_without_values(table, data, rows, cols, aggfunc, observed, margins_name)
        if not isinstance(marginal_result_set, tuple):
            return marginal_result_set
        result, margin_keys, row_margin = marginal_result_set
    row_margin = row_margin.reindex(result.columns, fill_value=fill_value)
    for k in margin_keys:
        if isinstance(k, str):
            row_margin[k] = grand_margin[k]
        else:
            row_margin[k] = grand_margin[k[0]]
    from pandas import DataFrame
    margin_dummy = DataFrame(row_margin, columns=Index([key])).T
    row_names = result.index.names
    for dtype in set(result.dtypes):
        if isinstance(dtype, ExtensionDtype):
            continue
        cols = result.select_dtypes([dtype]).columns
        margin_dummy[cols] = margin_dummy[cols].apply(maybe_downcast_to_dtype, args=(dtype,))
    result = result._append(margin_dummy)
    result.index.names = row_names
    return result