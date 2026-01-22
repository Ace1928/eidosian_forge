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
class _ScalarAccessIndexer(NDFrameIndexerBase):
    """
    Access scalars quickly.
    """
    _takeable: bool

    def _convert_key(self, key):
        raise AbstractMethodError(self)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            if not is_list_like_indexer(key):
                key = (key,)
            else:
                raise ValueError('Invalid call for scalar access (getting)!')
        key = self._convert_key(key)
        return self.obj._get_value(*key, takeable=self._takeable)

    def __setitem__(self, key, value) -> None:
        if isinstance(key, tuple):
            key = tuple((com.apply_if_callable(x, self.obj) for x in key))
        else:
            key = com.apply_if_callable(key, self.obj)
        if not isinstance(key, tuple):
            key = _tuplify(self.ndim, key)
        key = list(self._convert_key(key))
        if len(key) != self.ndim:
            raise ValueError('Not enough indexers for scalar access (setting)!')
        self.obj._set_value(*key, value=value, takeable=self._takeable)