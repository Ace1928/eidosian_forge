from typing import Any
import dask.dataframe as dd
from dask.distributed import Client
from fugue import DataFrame
from fugue.dev import (
from fugue.plugins import (
from fugue_dask._utils import DASK_UTILS
from fugue_dask.dataframe import DaskDataFrame
from fugue_dask.execution_engine import DaskExecutionEngine
@fugue_annotated_param(DaskExecutionEngine)
class _DaskExecutionEngineParam(ExecutionEngineParam):
    pass