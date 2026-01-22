from typing import Any
import ray.data as rd
from triad import run_at_def
from fugue import DataFrame, register_execution_engine
from fugue.dev import (
from fugue.plugins import as_fugue_dataset, infer_execution_engine
from .dataframe import RayDataFrame
from .execution_engine import RayExecutionEngine
@fugue_annotated_param(RayExecutionEngine)
class _RayExecutionEngineParam(ExecutionEngineParam):
    pass