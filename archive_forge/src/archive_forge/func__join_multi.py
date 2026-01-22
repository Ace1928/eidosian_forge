from __future__ import annotations
from collections import abc
from datetime import datetime
import functools
from itertools import zip_longest
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import BlockValuesRefs
import pandas._libs.join as libjoin
from pandas._libs.lib import (
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.arrays import (
from pandas.core.arrays.string_ import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.missing import clean_reindex_fill_method
from pandas.core.ops import get_op_result_name
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.core.strings.accessor import StringMethods
from pandas.io.formats.printing import (
@final
def _join_multi(self, other: Index, how: JoinHow):
    from pandas.core.indexes.multi import MultiIndex
    from pandas.core.reshape.merge import restore_dropped_levels_multijoin
    self_names_list = list(com.not_none(*self.names))
    other_names_list = list(com.not_none(*other.names))
    self_names_order = self_names_list.index
    other_names_order = other_names_list.index
    self_names = set(self_names_list)
    other_names = set(other_names_list)
    overlap = self_names & other_names
    if not overlap:
        raise ValueError('cannot join with no overlapping index names')
    if isinstance(self, MultiIndex) and isinstance(other, MultiIndex):
        ldrop_names = sorted(self_names - overlap, key=self_names_order)
        rdrop_names = sorted(other_names - overlap, key=other_names_order)
        if not len(ldrop_names + rdrop_names):
            self_jnlevels = self
            other_jnlevels = other.reorder_levels(self.names)
        else:
            self_jnlevels = self.droplevel(ldrop_names)
            other_jnlevels = other.droplevel(rdrop_names)
        join_idx, lidx, ridx = self_jnlevels.join(other_jnlevels, how=how, return_indexers=True)
        dropped_names = ldrop_names + rdrop_names
        levels, codes, names = restore_dropped_levels_multijoin(self, other, dropped_names, join_idx, lidx, ridx)
        multi_join_idx = MultiIndex(levels=levels, codes=codes, names=names, verify_integrity=False)
        multi_join_idx = multi_join_idx.remove_unused_levels()
        if how == 'right':
            level_order = other_names_list + ldrop_names
        else:
            level_order = self_names_list + rdrop_names
        multi_join_idx = multi_join_idx.reorder_levels(level_order)
        return (multi_join_idx, lidx, ridx)
    jl = next(iter(overlap))
    flip_order = False
    if isinstance(self, MultiIndex):
        self, other = (other, self)
        flip_order = True
        flip: dict[JoinHow, JoinHow] = {'right': 'left', 'left': 'right'}
        how = flip.get(how, how)
    level = other.names.index(jl)
    result = self._join_level(other, level, how=how)
    if flip_order:
        return (result[0], result[2], result[1])
    return result