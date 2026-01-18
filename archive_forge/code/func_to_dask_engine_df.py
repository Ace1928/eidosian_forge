import logging
import os
from typing import Any, Callable, Dict, List, Optional, Type, Union
import dask.dataframe as dd
import pandas as pd
from distributed import Client
from triad.collections import Schema
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.utils.pandas_like import PandasUtils
from triad.utils.threading import RunOnce
from triad.utils.io import makedirs
from fugue import StructuredRawSQL
from fugue.collections.partition import (
from fugue.constants import KEYWORD_PARALLELISM, KEYWORD_ROWCOUNT
from fugue.dataframe import (
from fugue.dataframe.utils import get_join_schemas
from fugue.exceptions import FugueBug
from fugue.execution.execution_engine import ExecutionEngine, MapEngine, SQLEngine
from fugue.execution.native_execution_engine import NativeExecutionEngine
from fugue_dask._constants import FUGUE_DASK_DEFAULT_CONF
from fugue_dask._io import load_df, save_df
from fugue_dask._utils import (
from fugue_dask.dataframe import DaskDataFrame
from ._constants import FUGUE_DASK_USE_ARROW
def to_dask_engine_df(df: Any, schema: Any=None) -> DaskDataFrame:
    """Convert a data structure to :class:`~.fugue_dask.dataframe.DaskDataFrame`

    :param data: :class:`~.fugue.dataframe.dataframe.DataFrame`,
      :class:`dask:dask.dataframe.DataFrame`,
      pandas DataFrame or list or iterable of arrays
    :param schema: |SchemaLikeObject|, defaults to None.
    :return: engine compatible dataframe

    .. note::

        * if the input is already :class:`~fugue_dask.dataframe.DaskDataFrame`,
          it should return itself
        * For list or iterable of arrays, ``schema`` must be specified
        * When ``schema`` is not None, a potential type cast may happen to ensure
          the dataframe's schema.
        * all other methods in the engine can take arbitrary dataframes and
          call this method to convert before doing anything
    """
    if isinstance(df, DataFrame):
        assert_or_throw(schema is None, ValueError('schema must be None when df is a DataFrame'))
        if isinstance(df, DaskDataFrame):
            return df
        if isinstance(df, PandasDataFrame):
            res = DaskDataFrame(df.native, df.schema)
        else:
            res = DaskDataFrame(df.as_array(type_safe=True), df.schema)
        res.reset_metadata(df.metadata)
        return res
    return DaskDataFrame(df, schema)