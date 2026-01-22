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
def _dummy_numpy_dispatcher(*arg_names: Literal['dtype', 'out'], deprecated: bool=False) -> Callable[[F], F]:
    """Decorator to handle the out= and dtype= keyword arguments.

    These parameters are deprecated in all dask.dataframe reduction methods
    and will be soon completely disallowed.
    However, these methods must continue accepting 'out=None' and/or 'dtype=None'
    indefinitely in order to support numpy dispatchers. For example,
    ``np.mean(df)`` calls ``df.mean(out=None, dtype=None)``.

    Parameters
    ----------
    deprecated: bool
        If True, warn if not None and then pass the parameter to the wrapped function
        If False, raise error if not None; do not pass the parameter down.

    See Also
    --------
    _deprecated_kwarg
    """

    def decorator(func: F) -> F:

        @wraps(func)
        def wrapper(*args, **kwargs):
            for name in arg_names:
                if deprecated:
                    if kwargs.get(name, None) is not None:
                        warnings.warn(f"the '{name}' keyword is deprecated and will be removed in a future version.", FutureWarning, stacklevel=2)
                elif kwargs.pop(name, None) is not None:
                    raise ValueError(f"the '{name}' keyword is not supported")
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator