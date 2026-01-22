import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from uuid import uuid4
import pandas as pd
import pyarrow as pa
import pyspark.sql as ps
from py4j.protocol import Py4JError
from pyspark import StorageLevel
from pyspark.rdd import RDD
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast, col, lit, row_number
from pyspark.sql.window import Window
from triad import IndexedOrderedDict, ParamDict, Schema, SerializableRLock
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.hash import to_uuid
from triad.utils.iter import EmptyAwareIterable
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import get_alter_func
from triad.utils.threading import RunOnce
from fugue import StructuredRawSQL
from fugue.collections.partition import (
from fugue.constants import KEYWORD_PARALLELISM, KEYWORD_ROWCOUNT
from fugue.dataframe import (
from fugue.dataframe.utils import get_join_schemas
from fugue.exceptions import FugueDataFrameInitError
from fugue.execution.execution_engine import ExecutionEngine, MapEngine, SQLEngine
from ._constants import FUGUE_SPARK_CONF_USE_PANDAS_UDF, FUGUE_SPARK_DEFAULT_CONF
from ._utils.convert import (
from ._utils.io import SparkIO
from ._utils.misc import is_spark_connect as _is_spark_connect
from ._utils.misc import is_spark_dataframe
from ._utils.partition import even_repartition, hash_repartition, rand_repartition
from .dataframe import SparkDataFrame
class SparkSQLEngine(SQLEngine):
    """`Spark SQL <https://spark.apache.org/sql/>`_ execution implementation.

    :param execution_engine: it must be :class:`~.SparkExecutionEngine`
    :raises ValueError: if the engine is not :class:`~.SparkExecutionEngine`
    """

    @property
    def dialect(self) -> Optional[str]:
        return 'spark'

    @property
    def execution_engine_constraint(self) -> Type[ExecutionEngine]:
        return SparkExecutionEngine

    @property
    def is_distributed(self) -> bool:
        return True

    def select(self, dfs: DataFrames, statement: StructuredRawSQL) -> DataFrame:
        _map: Dict[str, str] = {}
        for k, v in dfs.items():
            df = self.execution_engine._to_spark_df(v, create_view=True)
            _map[k] = df.alias
        _sql = statement.construct(_map, dialect=self.dialect, log=self.log)
        return SparkDataFrame(self.execution_engine.spark_session.sql(_sql))