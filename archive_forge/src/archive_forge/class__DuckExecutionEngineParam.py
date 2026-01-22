from typing import Any
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from triad import run_at_def
from fugue import (
from fugue.dev import (
from fugue.plugins import infer_execution_engine
from fugue_duckdb.dataframe import DuckDataFrame
from fugue_duckdb.execution_engine import DuckDBEngine, DuckExecutionEngine
@fugue_annotated_param(DuckExecutionEngine)
class _DuckExecutionEngineParam(ExecutionEngineParam):
    pass