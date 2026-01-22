from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import pyarrow as pa
from triad.collections.schema import Schema
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import cast_pa_table, pa_table_to_pandas
from fugue.dataset.api import (
from fugue.exceptions import FugueDataFrameOperationError
from .api import (
from .dataframe import DataFrame, LocalBoundedDataFrame, _input_schema
from .utils import (
@drop_columns.candidate(lambda df, *args, **kwargs: isinstance(df, pa.Table))
def _drop_pa_columns(df: pa.Table, columns: List[str]) -> pa.Table:
    cols = [x for x in df.schema.names if x not in columns]
    if len(cols) == 0:
        raise FugueDataFrameOperationError('cannot drop all columns')
    if len(cols) + len(columns) != len(df.columns):
        _assert_no_missing(df, columns)
    return df.select(cols)