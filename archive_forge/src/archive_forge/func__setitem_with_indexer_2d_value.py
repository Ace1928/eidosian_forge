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
def _setitem_with_indexer_2d_value(self, indexer, value):
    pi = indexer[0]
    ilocs = self._ensure_iterable_column_indexer(indexer[1])
    if not is_array_like(value):
        value = np.array(value, dtype=object)
    if len(ilocs) != value.shape[1]:
        raise ValueError('Must have equal len keys and value when setting with an ndarray')
    for i, loc in enumerate(ilocs):
        value_col = value[:, i]
        if is_object_dtype(value_col.dtype):
            value_col = value_col.tolist()
        self._setitem_single_column(loc, value_col, pi)