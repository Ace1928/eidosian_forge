from __future__ import annotations
import datetime
import warnings
from numbers import Integral
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pandas.core.window import Rolling as pd_Rolling
from dask.array.core import normalize_arg
from dask.base import tokenize
from dask.blockwise import BlockwiseDepDict
from dask.dataframe import methods
from dask.dataframe._compat import check_axis_keyword_deprecation
from dask.dataframe.core import (
from dask.dataframe.io import from_pandas
from dask.dataframe.multi import _maybe_align_partitions
from dask.dataframe.utils import (
from dask.delayed import unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import M, apply, derived_from, funcname, has_keyword
def _combined_parts(prev_part, current_part, next_part, before, after):
    msg = 'Partition size is less than overlapping window size. Try using ``df.repartition`` to increase the partition size.'
    if prev_part is not None and isinstance(before, Integral):
        if prev_part.shape[0] != before:
            raise NotImplementedError(msg)
    if next_part is not None and isinstance(after, Integral):
        if next_part.shape[0] != after:
            raise NotImplementedError(msg)
    parts = [p for p in (prev_part, current_part, next_part) if p is not None]
    combined = methods.concat(parts)
    return CombinedOutput((combined, len(prev_part) if prev_part is not None else None, len(next_part) if next_part is not None else None))