from typing import Any
import dask.dataframe as dd
from dask.distributed import Client
from fugue import DataFrame
from fugue.dev import (
from fugue.plugins import (
from fugue_dask._utils import DASK_UTILS
from fugue_dask.dataframe import DaskDataFrame
from fugue_dask.execution_engine import DaskExecutionEngine
@infer_execution_engine.candidate(lambda objs: is_pandas_or(objs, (dd.DataFrame, DaskDataFrame)))
def _infer_dask_client(objs: Any) -> Any:
    return DASK_UTILS.get_or_create_client()