from __future__ import annotations
from collections import abc
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.indexes.api import (
from pandas.core.internals import concatenate_managers
def _make_concat_multiindex(indexes, keys, levels=None, names=None) -> MultiIndex:
    if levels is None and isinstance(keys[0], tuple) or (levels is not None and len(levels) > 1):
        zipped = list(zip(*keys))
        if names is None:
            names = [None] * len(zipped)
        if levels is None:
            _, levels = factorize_from_iterables(zipped)
        else:
            levels = [ensure_index(x) for x in levels]
    else:
        zipped = [keys]
        if names is None:
            names = [None]
        if levels is None:
            levels = [ensure_index(keys).unique()]
        else:
            levels = [ensure_index(x) for x in levels]
    for level in levels:
        if not level.is_unique:
            raise ValueError(f'Level values not unique: {level.tolist()}')
    if not all_indexes_same(indexes) or not all((level.is_unique for level in levels)):
        codes_list = []
        for hlevel, level in zip(zipped, levels):
            to_concat = []
            if isinstance(hlevel, Index) and hlevel.equals(level):
                lens = [len(idx) for idx in indexes]
                codes_list.append(np.repeat(np.arange(len(hlevel)), lens))
            else:
                for key, index in zip(hlevel, indexes):
                    mask = isna(level) & isna(key) | (level == key)
                    if not mask.any():
                        raise ValueError(f'Key {key} not in level {level}')
                    i = np.nonzero(mask)[0][0]
                    to_concat.append(np.repeat(i, len(index)))
                codes_list.append(np.concatenate(to_concat))
        concat_index = _concat_indexes(indexes)
        if isinstance(concat_index, MultiIndex):
            levels.extend(concat_index.levels)
            codes_list.extend(concat_index.codes)
        else:
            codes, categories = factorize_from_iterable(concat_index)
            levels.append(categories)
            codes_list.append(codes)
        if len(names) == len(levels):
            names = list(names)
        else:
            if not len({idx.nlevels for idx in indexes}) == 1:
                raise AssertionError('Cannot concat indices that do not have the same number of levels')
            names = list(names) + list(get_unanimous_names(*indexes))
        return MultiIndex(levels=levels, codes=codes_list, names=names, verify_integrity=False)
    new_index = indexes[0]
    n = len(new_index)
    kpieces = len(indexes)
    new_names = list(names)
    new_levels = list(levels)
    new_codes = []
    for hlevel, level in zip(zipped, levels):
        hlevel_index = ensure_index(hlevel)
        mapped = level.get_indexer(hlevel_index)
        mask = mapped == -1
        if mask.any():
            raise ValueError(f'Values not found in passed level: {hlevel_index[mask]!s}')
        new_codes.append(np.repeat(mapped, n))
    if isinstance(new_index, MultiIndex):
        new_levels.extend(new_index.levels)
        new_codes.extend([np.tile(lab, kpieces) for lab in new_index.codes])
    else:
        new_levels.append(new_index.unique())
        single_codes = new_index.unique().get_indexer(new_index)
        new_codes.append(np.tile(single_codes, kpieces))
    if len(new_names) < len(new_levels):
        new_names.extend(new_index.names)
    return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)