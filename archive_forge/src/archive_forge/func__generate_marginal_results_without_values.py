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
def _generate_marginal_results_without_values(table: DataFrame, data: DataFrame, rows, cols, aggfunc, observed: bool, margins_name: Hashable='All'):
    margin_keys: list | Index
    if len(cols) > 0:
        margin_keys = []

        def _all_key():
            if len(cols) == 1:
                return margins_name
            return (margins_name,) + ('',) * (len(cols) - 1)
        if len(rows) > 0:
            margin = data.groupby(rows, observed=observed)[rows].apply(aggfunc)
            all_key = _all_key()
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)
        else:
            margin = data.groupby(level=0, axis=0, observed=observed).apply(aggfunc)
            all_key = _all_key()
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)
            return result
    else:
        result = table
        margin_keys = table.columns
    if len(cols):
        row_margin = data.groupby(cols, observed=observed)[cols].apply(aggfunc)
    else:
        row_margin = Series(np.nan, index=result.columns)
    return (result, margin_keys, row_margin)