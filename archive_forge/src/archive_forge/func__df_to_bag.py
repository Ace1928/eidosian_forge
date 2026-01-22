from __future__ import annotations
from collections.abc import Iterable
from functools import partial
from math import ceil
from operator import getitem
from threading import Lock
from typing import TYPE_CHECKING, Literal, overload
import numpy as np
import pandas as pd
import dask.array as da
from dask.base import is_dask_collection, tokenize
from dask.blockwise import BlockwiseDepDict, blockwise
from dask.dataframe._compat import is_any_real_numeric_dtype
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.core import (
from dask.dataframe.dispatch import meta_lib_from_array
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.dataframe.utils import (
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameIOLayer
from dask.utils import M, funcname, is_arraylike
def _df_to_bag(df, index=False, format='tuple'):
    if isinstance(df, pd.DataFrame):
        if format == 'tuple':
            return list(map(tuple, df.itertuples(index)))
        elif format == 'dict':
            if index:
                return [{**{'index': idx}, **values} for values, idx in zip(df.to_dict('records'), df.index)]
            else:
                return df.to_dict(orient='records')
    elif isinstance(df, pd.Series):
        if format == 'tuple':
            return list(df.items()) if index else list(df)
        elif format == 'dict':
            return df.to_frame().to_dict(orient='records')