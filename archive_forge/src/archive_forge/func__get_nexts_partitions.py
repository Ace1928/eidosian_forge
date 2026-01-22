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
def _get_nexts_partitions(df, after):
    """
    Helper to get the nexts partitions required for the overlap
    """
    dsk = {}
    df_name = df._name
    timedelta_partition_message = 'Partition size is less than specified window. Try using ``df.repartition`` to increase the partition size'
    name_b = 'overlap-append-' + tokenize(df, after)
    if after and isinstance(after, Integral):
        nexts = []
        for i in range(1, df.npartitions):
            key = (name_b, i)
            dsk[key] = (M.head, (df_name, i), after)
            nexts.append(key)
        nexts.append(None)
    elif isinstance(after, datetime.timedelta):
        deltas = pd.Series(df.divisions).diff().iloc[1:-1]
        if (after > deltas).any():
            raise ValueError(timedelta_partition_message)
        nexts = []
        for i in range(1, df.npartitions):
            key = (name_b, i)
            dsk[key] = (_head_timedelta, (df_name, i - 0), (df_name, i), after)
            nexts.append(key)
        nexts.append(None)
    else:
        nexts = [None] * df.npartitions
    return (dsk, nexts)