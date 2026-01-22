from __future__ import annotations
from contextlib import suppress
import sys
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs.indexing import NDFrameIndexerBase
from pandas._libs.lib import item_from_zerodim
from pandas.compat import PYPY
from pandas.errors import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import algorithms as algos
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.api import (
def _setitem_with_indexer(self, indexer, value, name: str='iloc'):
    """
        _setitem_with_indexer is for setting values on a Series/DataFrame
        using positional indexers.

        If the relevant keys are not present, the Series/DataFrame may be
        expanded.

        This method is currently broken when dealing with non-unique Indexes,
        since it goes from positional indexers back to labels when calling
        BlockManager methods, see GH#12991, GH#22046, GH#15686.
        """
    info_axis = self.obj._info_axis_number
    take_split_path = not self.obj._mgr.is_single_block
    if not take_split_path and isinstance(value, ABCDataFrame):
        take_split_path = not value._mgr.is_single_block
    if not take_split_path and len(self.obj._mgr.arrays) and (self.ndim > 1):
        val = list(value.values()) if isinstance(value, dict) else value
        arr = self.obj._mgr.arrays[0]
        take_split_path = not can_hold_element(arr, extract_array(val, extract_numpy=True))
    if isinstance(indexer, tuple) and len(indexer) == len(self.obj.axes):
        for i, ax in zip(indexer, self.obj.axes):
            if isinstance(ax, MultiIndex) and (not (is_integer(i) or com.is_null_slice(i))):
                take_split_path = True
                break
    if isinstance(indexer, tuple):
        nindexer = []
        for i, idx in enumerate(indexer):
            if isinstance(idx, dict):
                key, _ = convert_missing_indexer(idx)
                if self.ndim > 1 and i == info_axis:
                    if not len(self.obj):
                        if not is_list_like_indexer(value):
                            raise ValueError('cannot set a frame with no defined index and a scalar')
                        self.obj[key] = value
                        return
                    if com.is_null_slice(indexer[0]):
                        self.obj[key] = value
                        return
                    elif is_array_like(value):
                        arr = extract_array(value, extract_numpy=True)
                        taker = -1 * np.ones(len(self.obj), dtype=np.intp)
                        empty_value = algos.take_nd(arr, taker)
                        if not isinstance(value, ABCSeries):
                            if isinstance(arr, np.ndarray) and arr.ndim == 1 and (len(arr) == 1):
                                arr = arr[0, ...]
                            empty_value[indexer[0]] = arr
                            self.obj[key] = empty_value
                            return
                        self.obj[key] = empty_value
                    elif not is_list_like(value):
                        self.obj[key] = construct_1d_array_from_inferred_fill_value(value, len(self.obj))
                    else:
                        self.obj[key] = infer_fill_value(value)
                    new_indexer = convert_from_missing_indexer_tuple(indexer, self.obj.axes)
                    self._setitem_with_indexer(new_indexer, value, name)
                    return
                index = self.obj._get_axis(i)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'The behavior of Index.insert with object-dtype is deprecated', category=FutureWarning)
                    labels = index.insert(len(index), key)
                taker = np.arange(len(index) + 1, dtype=np.intp)
                taker[-1] = -1
                reindexers = {i: (labels, taker)}
                new_obj = self.obj._reindex_with_indexers(reindexers, allow_dups=True)
                self.obj._mgr = new_obj._mgr
                self.obj._maybe_update_cacher(clear=True)
                self.obj._is_copy = None
                nindexer.append(labels.get_loc(key))
            else:
                nindexer.append(idx)
        indexer = tuple(nindexer)
    else:
        indexer, missing = convert_missing_indexer(indexer)
        if missing:
            self._setitem_with_indexer_missing(indexer, value)
            return
    if name == 'loc':
        indexer, value = self._maybe_mask_setitem_value(indexer, value)
    if take_split_path:
        self._setitem_with_indexer_split_path(indexer, value, name)
    else:
        self._setitem_single_block(indexer, value, name)