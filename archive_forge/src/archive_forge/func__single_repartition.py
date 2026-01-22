from typing import Any, Iterable, List
import pyspark.sql as ps
import pyspark.sql.functions as psf
from pyspark import RDD
from pyspark.sql import SparkSession
import warnings
from .convert import to_schema, to_spark_schema
from .misc import is_spark_connect
def _single_repartition(df: ps.DataFrame) -> ps.DataFrame:
    return df.withColumn(_PARTITION_DUMMY_KEY, psf.lit(0)).repartition(_PARTITION_DUMMY_KEY).drop(_PARTITION_DUMMY_KEY)