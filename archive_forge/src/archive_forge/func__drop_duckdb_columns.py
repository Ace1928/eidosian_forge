from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import pyarrow as pa
from duckdb import DuckDBPyRelation
from triad import Schema, assert_or_throw
from triad.utils.pyarrow import LARGE_TYPES_REPLACEMENT, replace_types_in_table
from fugue import ArrowDataFrame, DataFrame, LocalBoundedDataFrame
from fugue.dataframe.arrow_dataframe import _pa_table_as_pandas
from fugue.dataframe.utils import (
from fugue.exceptions import FugueDataFrameOperationError, FugueDatasetEmptyError
from fugue.plugins import (
from ._utils import encode_column_name, to_duck_type, to_pa_type
@drop_columns.candidate(lambda df, *args, **kwargs: isinstance(df, DuckDBPyRelation))
def _drop_duckdb_columns(df: DuckDBPyRelation, columns: List[str]) -> DuckDBPyRelation:
    _columns = {c: 1 for c in columns}
    cols = [col for col in df.columns if _columns.pop(col, None) is None]
    assert_or_throw(len(cols) > 0, FugueDataFrameOperationError('must keep at least one column'))
    assert_or_throw(len(_columns) == 0, FugueDataFrameOperationError('found nonexistent columns {_columns}'))
    return df.project(','.join((encode_column_name(n) for n in cols)))