from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import pyarrow as pa
import pyspark.sql as ps
from pyspark.sql.functions import col
from triad import SerializableRLock
from triad.collections.schema import SchemaError
from triad.utils.assertion import assert_or_throw
from fugue.dataframe import (
from fugue.dataframe.utils import pa_table_as_array, pa_table_as_dicts
from fugue.exceptions import FugueDataFrameOperationError
from fugue.plugins import (
from ._utils.convert import (
from ._utils.misc import is_spark_connect, is_spark_dataframe
@rename.candidate(lambda df, *args, **kwargs: is_spark_dataframe(df))
def _rename_spark_df(df: ps.DataFrame, columns: Dict[str, Any], as_fugue: bool=False) -> ps.DataFrame:
    if len(columns) == 0:
        return df
    _assert_no_missing(df, columns.keys())
    return _adjust_df(_rename_spark_dataframe(df, columns), as_fugue=as_fugue)