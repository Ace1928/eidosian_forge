from typing import Any, Iterable, List
import pyspark.sql as ps
import pyspark.sql.functions as psf
from pyspark import RDD
from pyspark.sql import SparkSession
import warnings
from .convert import to_schema, to_spark_schema
from .misc import is_spark_connect
def _to_kv(rows: Iterable[Any]) -> Iterable[Any]:
    for row in rows:
        yield (row[0], row[1:])