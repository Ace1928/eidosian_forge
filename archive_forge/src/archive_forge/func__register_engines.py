from typing import Any, Optional, Tuple
import pyspark.rdd as pr
import pyspark.sql as ps
from pyspark import SparkContext
from pyspark.sql import SparkSession
from triad import run_at_def
from fugue import DataFrame, ExecutionEngine, register_execution_engine
from fugue.dev import (
from fugue.extensions import namespace_candidate
from fugue.plugins import as_fugue_dataset, infer_execution_engine, parse_creator
from fugue_spark.dataframe import SparkDataFrame
from fugue_spark.execution_engine import SparkExecutionEngine
from ._utils.misc import SparkConnectDataFrame, SparkConnectSession, is_spark_dataframe
def _register_engines() -> None:
    register_execution_engine('spark', lambda conf, **kwargs: SparkExecutionEngine(conf=conf), on_dup='ignore')
    register_execution_engine(SparkSession, lambda session, conf, **kwargs: SparkExecutionEngine(session, conf=conf), on_dup='ignore')
    if SparkConnectSession is not None:
        register_execution_engine(SparkConnectSession, lambda session, conf, **kwargs: SparkExecutionEngine(session, conf=conf), on_dup='ignore')