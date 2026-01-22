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
@final
def _ensure_listlike_indexer(self, key, axis=None, value=None) -> None:
    """
        Ensure that a list-like of column labels are all present by adding them if
        they do not already exist.

        Parameters
        ----------
        key : list-like of column labels
            Target labels.
        axis : key axis if known
        """
    column_axis = 1
    if self.ndim != 2:
        return
    if isinstance(key, tuple) and len(key) > 1:
        key = key[column_axis]
        axis = column_axis
    if axis == column_axis and (not isinstance(self.obj.columns, MultiIndex)) and is_list_like_indexer(key) and (not com.is_bool_indexer(key)) and all((is_hashable(k) for k in key)):
        keys = self.obj.columns.union(key, sort=False)
        diff = Index(key).difference(self.obj.columns, sort=False)
        if len(diff):
            indexer = np.arange(len(keys), dtype=np.intp)
            indexer[len(self.obj.columns):] = -1
            new_mgr = self.obj._mgr.reindex_indexer(keys, indexer=indexer, axis=0, only_slice=True, use_na_proxy=True)
            self.obj._mgr = new_mgr
            return
        self.obj._mgr = self.obj._mgr.reindex_axis(keys, axis=0, only_slice=True)