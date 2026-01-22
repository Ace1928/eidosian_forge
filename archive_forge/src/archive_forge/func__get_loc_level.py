from __future__ import annotations
from collections.abc import (
from functools import wraps
from sys import getsizeof
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import (
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.construction import sanitize_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.io.formats.printing import (
def _get_loc_level(self, key, level: int | list[int]=0):
    """
        get_loc_level but with `level` known to be positional, not name-based.
        """

    def maybe_mi_droplevels(indexer, levels):
        """
            If level does not exist or all levels were dropped, the exception
            has to be handled outside.
            """
        new_index = self[indexer]
        for i in sorted(levels, reverse=True):
            new_index = new_index._drop_level_numbers([i])
        return new_index
    if isinstance(level, (tuple, list)):
        if len(key) != len(level):
            raise AssertionError('Key for location must have same length as number of levels')
        result = None
        for lev, k in zip(level, key):
            loc, new_index = self._get_loc_level(k, level=lev)
            if isinstance(loc, slice):
                mask = np.zeros(len(self), dtype=bool)
                mask[loc] = True
                loc = mask
            result = loc if result is None else result & loc
        try:
            mi = maybe_mi_droplevels(result, level)
        except ValueError:
            mi = self[result]
        return (result, mi)
    if isinstance(key, list):
        key = tuple(key)
    if isinstance(key, tuple) and level == 0:
        try:
            if key in self.levels[0]:
                indexer = self._get_level_indexer(key, level=level)
                new_index = maybe_mi_droplevels(indexer, [0])
                return (indexer, new_index)
        except (TypeError, InvalidIndexError):
            pass
        if not any((isinstance(k, slice) for k in key)):
            if len(key) == self.nlevels and self.is_unique:
                try:
                    return (self._engine.get_loc(key), None)
                except KeyError as err:
                    raise KeyError(key) from err
                except TypeError:
                    pass
            indexer = self.get_loc(key)
            ilevels = [i for i in range(len(key)) if key[i] != slice(None, None)]
            if len(ilevels) == self.nlevels:
                if is_integer(indexer):
                    return (indexer, None)
                ilevels = [i for i in range(len(key)) if (not isinstance(key[i], str) or not self.levels[i]._supports_partial_string_indexing) and key[i] != slice(None, None)]
                if len(ilevels) == self.nlevels:
                    ilevels = []
            return (indexer, maybe_mi_droplevels(indexer, ilevels))
        else:
            indexer = None
            for i, k in enumerate(key):
                if not isinstance(k, slice):
                    loc_level = self._get_level_indexer(k, level=i)
                    if isinstance(loc_level, slice):
                        if com.is_null_slice(loc_level) or com.is_full_slice(loc_level, len(self)):
                            continue
                        k_index = np.zeros(len(self), dtype=bool)
                        k_index[loc_level] = True
                    else:
                        k_index = loc_level
                elif com.is_null_slice(k):
                    continue
                else:
                    raise TypeError(f'Expected label or tuple of labels, got {key}')
                if indexer is None:
                    indexer = k_index
                else:
                    indexer &= k_index
            if indexer is None:
                indexer = slice(None, None)
            ilevels = [i for i in range(len(key)) if key[i] != slice(None, None)]
            return (indexer, maybe_mi_droplevels(indexer, ilevels))
    else:
        indexer = self._get_level_indexer(key, level=level)
        if isinstance(key, str) and self.levels[level]._supports_partial_string_indexing:
            check = self.levels[level].get_loc(key)
            if not is_integer(check):
                return (indexer, self[indexer])
        try:
            result_index = maybe_mi_droplevels(indexer, [level])
        except ValueError:
            result_index = self[indexer]
        return (indexer, result_index)