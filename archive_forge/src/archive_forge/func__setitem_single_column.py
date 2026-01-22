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
def _setitem_single_column(self, loc: int, value, plane_indexer) -> None:
    """

        Parameters
        ----------
        loc : int
            Indexer for column position
        plane_indexer : int, slice, listlike[int]
            The indexer we use for setitem along axis=0.
        """
    pi = plane_indexer
    is_full_setter = com.is_null_slice(pi) or com.is_full_slice(pi, len(self.obj))
    is_null_setter = com.is_empty_slice(pi) or (is_array_like(pi) and len(pi) == 0)
    if is_null_setter:
        return
    elif is_full_setter:
        try:
            self.obj._mgr.column_setitem(loc, plane_indexer, value, inplace_only=True)
        except (ValueError, TypeError, LossySetitemError):
            dtype = self.obj.dtypes.iloc[loc]
            if dtype not in (np.void, object) and (not self.obj.empty):
                warnings.warn(f"Setting an item of incompatible dtype is deprecated and will raise in a future error of pandas. Value '{value}' has dtype incompatible with {dtype}, please explicitly cast to a compatible dtype first.", FutureWarning, stacklevel=find_stack_level())
            self.obj.isetitem(loc, value)
    else:
        dtype = self.obj.dtypes.iloc[loc]
        if dtype == np.void:
            self.obj.iloc[:, loc] = construct_1d_array_from_inferred_fill_value(value, len(self.obj))
        self.obj._mgr.column_setitem(loc, plane_indexer, value)
    self.obj._clear_item_cache()