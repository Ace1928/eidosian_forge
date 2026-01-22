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
def _getitem_lowerdim(self, tup: tuple):
    if self.axis is not None:
        axis = self.obj._get_axis_number(self.axis)
        return self._getitem_axis(tup, axis=axis)
    if self._is_nested_tuple_indexer(tup):
        return self._getitem_nested_tuple(tup)
    ax0 = self.obj._get_axis(0)
    if isinstance(ax0, MultiIndex) and self.name != 'iloc' and (not any((isinstance(x, slice) for x in tup))):
        with suppress(IndexingError):
            return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(tup)
    tup = self._validate_key_length(tup)
    for i, key in enumerate(tup):
        if is_label_like(key):
            section = self._getitem_axis(key, axis=i)
            if section.ndim == self.ndim:
                new_key = tup[:i] + (_NS,) + tup[i + 1:]
            else:
                new_key = tup[:i] + tup[i + 1:]
                if len(new_key) == 1:
                    new_key = new_key[0]
            if com.is_null_slice(new_key):
                return section
            return getattr(section, self.name)[new_key]
    raise IndexingError('not applicable')