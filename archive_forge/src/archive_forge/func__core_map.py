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
def _core_map(pdf: pd.DataFrame) -> pd.DataFrame:
    if len(partition_spec.presort) > 0:
        pdf = pdf.sort_values(presort_keys, ascending=presort_asc)
    input_df = PandasDataFrame(pdf.reset_index(drop=True), input_schema, pandas_df_wrapper=True)
    if on_init_once is not None:
        on_init_once(0, input_df)
    cursor.set(lambda: input_df.peek_array(), 0, 0)
    output_df = map_func(cursor, input_df)
    return output_df.as_pandas()[output_schema.names]