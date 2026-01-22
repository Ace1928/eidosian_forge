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
class _Mapper(object):

    def __init__(self, df: DataFrame, map_func: Callable[[PartitionCursor, LocalDataFrame], LocalDataFrame], output_schema: Any, partition_spec: PartitionSpec, on_init: Optional[Callable[[int, DataFrame], Any]]):
        super().__init__()
        self.schema = df.schema
        self.output_schema = Schema(output_schema)
        self.partition_spec = partition_spec
        self.map_func = map_func
        self.on_init = on_init
        self.cursor = self.partition_spec.get_cursor(self.schema, 0)

    def run(self, no: int, rows: Iterable[ps.Row]) -> Iterable[Any]:
        df = IterableDataFrame(to_type_safe_input(rows, self.schema), self.schema)
        if df.empty:
            return
        if self.on_init is not None:
            self.on_init(no, df)
        if self.partition_spec.empty or self.partition_spec.algo == 'coarse':
            partitions: Iterable[Tuple[int, int, EmptyAwareIterable]] = [(0, 0, df.native)]
        else:
            partitioner = self.partition_spec.get_partitioner(self.schema)
            partitions = partitioner.partition(df.native)
        for pn, sn, sub in partitions:
            self.cursor.set(sub.peek(), pn, sn)
            sub_df = IterableDataFrame(sub, self.schema)
            res = self.map_func(self.cursor, sub_df)
            for r in res.as_array_iterable(type_safe=True):
                yield r