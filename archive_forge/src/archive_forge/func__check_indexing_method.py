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
def _check_indexing_method(self, method: str_t | None, limit: int | None=None, tolerance=None) -> None:
    """
        Raise if we have a get_indexer `method` that is not supported or valid.
        """
    if method not in [None, 'bfill', 'backfill', 'pad', 'ffill', 'nearest']:
        raise ValueError('Invalid fill method')
    if self._is_multi:
        if method == 'nearest':
            raise NotImplementedError("method='nearest' not implemented yet for MultiIndex; see GitHub issue 9365")
        if method in ('pad', 'backfill'):
            if tolerance is not None:
                raise NotImplementedError('tolerance not implemented yet for MultiIndex')
    if isinstance(self.dtype, (IntervalDtype, CategoricalDtype)):
        if method is not None:
            raise NotImplementedError(f'method {method} not yet implemented for {type(self).__name__}')
    if method is None:
        if tolerance is not None:
            raise ValueError('tolerance argument only valid if doing pad, backfill or nearest reindexing')
        if limit is not None:
            raise ValueError('limit argument only valid if doing pad, backfill or nearest reindexing')