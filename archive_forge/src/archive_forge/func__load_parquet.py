from typing import Any, Callable, Dict, List, Optional, Union
import pyspark.sql as ps
from pyspark.sql import SparkSession
from triad.collections import Schema
from triad.collections.dict import ParamDict
from triad.utils.assertion import assert_or_throw
from fugue._utils.io import FileParser, save_df
from fugue.collections.partition import PartitionSpec
from fugue.dataframe import DataFrame, PandasDataFrame
from fugue_spark.dataframe import SparkDataFrame
from .convert import to_schema, to_spark_schema
def _load_parquet(self, p: List[str], columns: Any=None, **kwargs: Any) -> DataFrame:
    sdf = self._session.read.parquet(*p, **kwargs)
    if columns is None:
        return SparkDataFrame(sdf)
    if isinstance(columns, list):
        return SparkDataFrame(sdf)[columns]
    schema = Schema(columns)
    return SparkDataFrame(sdf[schema.names], schema)