from __future__ import annotations
import operator
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial, wraps
from numbers import Integral, Number
from operator import getitem
from pprint import pformat
from typing import Any, ClassVar, Literal, cast
import numpy as np
import pandas as pd
from pandas.api.types import (
from tlz import first, merge, partition_all, remove, unique
import dask.array as da
from dask import config, core
from dask.array.core import Array, normalize_arg
from dask.bag import map_partitions as map_bag_partitions
from dask.base import (
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise
from dask.context import globalmethod
from dask.dataframe import methods
from dask.dataframe._compat import (
from dask.dataframe.accessor import CachedAccessor, DatetimeAccessor, StringAccessor
from dask.dataframe.categorical import CategoricalAccessor, categorize
from dask.dataframe.dispatch import (
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import (
from dask.delayed import Delayed, delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameTreeReduction
from dask.typing import Graph, NestedKeys, no_default
from dask.utils import (
from dask.widgets import get_template
def _map_freq_to_period_start(freq):
    """Ensure that the frequency pertains to the **start** of a period.

    If e.g. `freq='M'`, then the divisions are:
        - 2021-31-1 00:00:00 (start of February partition)
        - 2021-2-28 00:00:00 (start of March partition)
        - ...

    but this **should** be:
        - 2021-2-1 00:00:00 (start of February partition)
        - 2021-3-1 00:00:00 (start of March partition)
        - ...

    Therefore, we map `freq='M'` to `freq='MS'` (same for quarter and year).
    """
    if not isinstance(freq, str):
        return freq
    offset = pd.tseries.frequencies.to_offset(freq)
    offset_type_name = type(offset).__name__
    if not offset_type_name.endswith('End'):
        return freq
    new_offset = offset_type_name[:-len('End')] + 'Begin'
    try:
        new_offset_type = getattr(pd.tseries.offsets, new_offset)
        if '-' in freq:
            _, anchor = freq.split('-')
            anchor = '-' + anchor
        else:
            anchor = ''
        n = str(offset.n) if offset.n != 1 else ''
        return f'{n}{new_offset_type._prefix}{anchor}'
    except AttributeError:
        return freq