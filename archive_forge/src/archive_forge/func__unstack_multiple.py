from __future__ import annotations
import itertools
from typing import (
import warnings
import numpy as np
import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.algorithms import (
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
def _unstack_multiple(data: Series | DataFrame, clocs, fill_value=None, sort: bool=True):
    if len(clocs) == 0:
        return data
    index = data.index
    index = cast(MultiIndex, index)
    if clocs in index.names:
        clocs = [clocs]
    clocs = [index._get_level_number(i) for i in clocs]
    rlocs = [i for i in range(index.nlevels) if i not in clocs]
    clevels = [index.levels[i] for i in clocs]
    ccodes = [index.codes[i] for i in clocs]
    cnames = [index.names[i] for i in clocs]
    rlevels = [index.levels[i] for i in rlocs]
    rcodes = [index.codes[i] for i in rlocs]
    rnames = [index.names[i] for i in rlocs]
    shape = tuple((len(x) for x in clevels))
    group_index = get_group_index(ccodes, shape, sort=False, xnull=False)
    comp_ids, obs_ids = compress_group_index(group_index, sort=False)
    recons_codes = decons_obs_group_ids(comp_ids, obs_ids, shape, ccodes, xnull=False)
    if not rlocs:
        dummy_index = Index(obs_ids, name='__placeholder__')
    else:
        dummy_index = MultiIndex(levels=rlevels + [obs_ids], codes=rcodes + [comp_ids], names=rnames + ['__placeholder__'], verify_integrity=False)
    if isinstance(data, Series):
        dummy = data.copy()
        dummy.index = dummy_index
        unstacked = dummy.unstack('__placeholder__', fill_value=fill_value, sort=sort)
        new_levels = clevels
        new_names = cnames
        new_codes = recons_codes
    else:
        if isinstance(data.columns, MultiIndex):
            result = data
            while clocs:
                val = clocs.pop(0)
                result = result.unstack(val, fill_value=fill_value, sort=sort)
                clocs = [v if v < val else v - 1 for v in clocs]
            return result
        dummy_df = data.copy(deep=False)
        dummy_df.index = dummy_index
        unstacked = dummy_df.unstack('__placeholder__', fill_value=fill_value, sort=sort)
        if isinstance(unstacked, Series):
            unstcols = unstacked.index
        else:
            unstcols = unstacked.columns
        assert isinstance(unstcols, MultiIndex)
        new_levels = [unstcols.levels[0]] + clevels
        new_names = [data.columns.name] + cnames
        new_codes = [unstcols.codes[0]]
        new_codes.extend((rec.take(unstcols.codes[-1]) for rec in recons_codes))
    new_columns = MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)
    if isinstance(unstacked, Series):
        unstacked.index = new_columns
    else:
        unstacked.columns = new_columns
    return unstacked