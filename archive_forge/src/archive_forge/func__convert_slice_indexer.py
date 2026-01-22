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
def _convert_slice_indexer(self, key: slice, kind: Literal['loc', 'getitem']):
    """
        Convert a slice indexer.

        By definition, these are labels unless 'iloc' is passed in.
        Floats are not allowed as the start, step, or stop of the slice.

        Parameters
        ----------
        key : label of the slice bound
        kind : {'loc', 'getitem'}
        """
    start, stop, step = (key.start, key.stop, key.step)
    is_index_slice = is_valid_positional_slice(key)
    if lib.is_np_dtype(self.dtype, 'f'):
        if kind == 'getitem' and is_index_slice and (not start == stop) and (step != 0):
            warnings.warn('The behavior of obj[i:j] with a float-dtype index is deprecated. In a future version, this will be treated as positional instead of label-based. For label-based slicing, use obj.loc[i:j] instead', FutureWarning, stacklevel=find_stack_level())
        return self.slice_indexer(start, stop, step)
    if kind == 'getitem':
        if is_index_slice:
            return key
        elif self.dtype.kind in 'iu':
            self._validate_indexer('slice', key.start, 'getitem')
            self._validate_indexer('slice', key.stop, 'getitem')
            self._validate_indexer('slice', key.step, 'getitem')
            return key
    is_positional = is_index_slice and self._should_fallback_to_positional
    if is_positional:
        try:
            if start is not None:
                self.get_loc(start)
            if stop is not None:
                self.get_loc(stop)
            is_positional = False
        except KeyError:
            pass
    if com.is_null_slice(key):
        indexer = key
    elif is_positional:
        if kind == 'loc':
            raise TypeError('Slicing a positional slice with .loc is not allowed, Use .loc with labels or .iloc with positions instead.')
        indexer = key
    else:
        indexer = self.slice_indexer(start, stop, step)
    return indexer