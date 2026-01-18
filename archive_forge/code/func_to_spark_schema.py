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
def to_spark_schema(obj: Any) -> pt.StructType:
    assert_arg_not_none(obj, 'schema')
    if isinstance(obj, pt.StructType):
        return obj
    if is_spark_dataframe(obj):
        return obj.schema
    return _from_arrow_schema(Schema(obj).pa_schema)