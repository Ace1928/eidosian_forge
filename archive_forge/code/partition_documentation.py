from typing import Any, Iterable, List
import pyspark.sql as ps
import pyspark.sql.functions as psf
from pyspark import RDD
from pyspark.sql import SparkSession
import warnings
from .convert import to_schema, to_spark_schema
from .misc import is_spark_connect

    Modified from
    https://github.com/davies/spark/blob/cebe5bfe263baf3349353f1473f097396821514a/python/pyspark/rdd.py

    