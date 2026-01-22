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
@head.candidate(lambda df, *args, **kwargs: is_spark_dataframe(df))
def _spark_df_head(df: ps.DataFrame, n: int, columns: Optional[List[str]]=None, as_fugue: bool=False) -> pd.DataFrame:
    if columns is not None:
        df = df[columns]
    res = df.limit(n)
    return SparkDataFrame(res).as_local() if as_fugue else to_pandas(res)