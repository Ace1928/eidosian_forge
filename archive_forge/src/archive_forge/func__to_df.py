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
def _to_df() -> SparkDataFrame:
    if isinstance(df, DataFrame):
        assert_or_throw(schema is None, ValueError('schema must be None when df is a DataFrame'))
        if isinstance(df, SparkDataFrame):
            return df
        if isinstance(df, (ArrayDataFrame, IterableDataFrame)):
            adf = ArrowDataFrame(df.as_array(type_safe=False), df.schema)
            sdf = to_spark_df(self.spark_session, adf, df.schema)
            return SparkDataFrame(sdf, df.schema)
        if any((pa.types.is_struct(t) for t in df.schema.types)):
            sdf = to_spark_df(self.spark_session, df.as_array(type_safe=True), df.schema)
        else:
            sdf = to_spark_df(self.spark_session, df, df.schema)
        return SparkDataFrame(sdf, df.schema)
    if is_spark_dataframe(df):
        return SparkDataFrame(df, None if schema is None else to_schema(schema))
    if isinstance(df, RDD):
        assert_arg_not_none(schema, 'schema')
        sdf = to_spark_df(self.spark_session, df, schema)
        return SparkDataFrame(sdf, to_schema(schema))
    if isinstance(df, pd.DataFrame):
        if PD_UTILS.empty(df):
            temp_schema = to_spark_schema(PD_UTILS.to_schema(df))
            sdf = to_spark_df(self.spark_session, [], temp_schema)
        else:
            sdf = to_spark_df(self.spark_session, df)
        return SparkDataFrame(sdf, schema)
    assert_or_throw(schema is not None, FugueDataFrameInitError("schema can't be None"))
    adf = ArrowDataFrame(df, to_schema(schema))
    map_pos = [i for i, t in enumerate(adf.schema.types) if pa.types.is_map(t)]
    if len(map_pos) == 0:
        sdf = to_spark_df(self.spark_session, adf.as_array(), adf.schema)
    else:

        def to_dict(rows: Iterable[List[Any]]) -> Iterable[List[Any]]:
            for row in rows:
                for p in map_pos:
                    row[p] = dict(row[p])
                yield row
        sdf = to_spark_df(self.spark_session, to_dict(adf.as_array_iterable()), adf.schema)
    return SparkDataFrame(sdf, adf.schema)