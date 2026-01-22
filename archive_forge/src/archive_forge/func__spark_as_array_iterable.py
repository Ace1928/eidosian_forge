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
@as_array_iterable.candidate(lambda df, *args, **kwargs: is_spark_dataframe(df))
def _spark_as_array_iterable(df: ps.DataFrame, columns: Optional[List[str]]=None, type_safe: bool=False) -> Iterable[Any]:
    if is_spark_connect(df):
        yield from _spark_as_array(df, columns, type_safe=type_safe)
    else:
        assert_or_throw(columns is None or len(columns) > 0, ValueError('empty columns'))
        _df = df if columns is None or len(columns) == 0 else df[columns]
        if not type_safe:
            for row in to_type_safe_input(_df.rdd.toLocalIterator(), to_schema(_df.schema)):
                yield list(row)
        else:
            tdf = IterableDataFrame(_spark_as_array_iterable(_df, type_safe=False), to_schema(_df.schema))
            yield from tdf.as_array_iterable(type_safe=True)