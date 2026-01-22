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
def _gp_map(pdf: pd.DataFrame) -> pd.DataFrame:
    if pdf.shape[0] == 0:
        return PandasDataFrame([], output_schema).as_pandas()
    pdf = pdf.reset_index(drop=True)
    pdf = _fix_dask_bug(pdf)
    pu = PandasUtils()
    res = pu.safe_groupby_apply(pdf, partition_spec.partition_by, _core_map)
    return res.astype(output_dtypes)