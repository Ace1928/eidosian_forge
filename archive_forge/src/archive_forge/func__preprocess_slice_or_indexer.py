from __future__ import annotations
from collections.abc import (
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.ops import (
def _preprocess_slice_or_indexer(slice_or_indexer: slice | np.ndarray, length: int, allow_fill: bool):
    if isinstance(slice_or_indexer, slice):
        return ('slice', slice_or_indexer, libinternals.slice_len(slice_or_indexer, length))
    else:
        if not isinstance(slice_or_indexer, np.ndarray) or slice_or_indexer.dtype.kind != 'i':
            dtype = getattr(slice_or_indexer, 'dtype', None)
            raise TypeError(type(slice_or_indexer), dtype)
        indexer = ensure_platform_int(slice_or_indexer)
        if not allow_fill:
            indexer = maybe_convert_indices(indexer, length)
        return ('fancy', indexer, len(indexer))