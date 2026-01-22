import pickle
from typing import Any, Iterable, List, Tuple, Optional
import pandas as pd
import pyarrow as pa
import pyspark
import pyspark.sql as ps
import pyspark.sql.types as pt
from packaging import version
from pyarrow.types import is_list, is_struct, is_timestamp
from pyspark.sql.pandas.types import (
from triad.collections import Schema
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP, cast_pa_table
from triad.utils.schema import quote_name
import fugue.api as fa
from fugue import DataFrame
from .misc import is_spark_dataframe
def _from_arrow_type(dt: pa.DataType) -> pt.DataType:
    if is_struct(dt):
        return pt.StructType([pt.StructField(field.name, _from_arrow_type(field.type), nullable=True) for field in dt])
    elif is_list(dt):
        if is_timestamp(dt.value_type):
            raise TypeError('Spark: unsupported type in conversion from Arrow: ' + str(dt))
        return pt.ArrayType(_from_arrow_type(dt.value_type))
    return from_arrow_type(dt)