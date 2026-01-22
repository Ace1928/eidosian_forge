import logging
from typing import Any, Dict, Iterable, List, Optional, Union
import duckdb
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from triad import SerializableRLock
from triad.utils.assertion import assert_or_throw
from triad.utils.schema import quote_name
from fugue import (
from fugue.collections.partition import PartitionSpec, parse_presort_exp
from fugue.collections.sql import StructuredRawSQL, TempTableName
from fugue.dataframe import DataFrame, DataFrames, LocalBoundedDataFrame
from fugue.dataframe.utils import get_join_schemas
from ._io import DuckDBIO
from ._utils import (
from .dataframe import DuckDataFrame, _duck_as_arrow
def _other_select(self, dfs: DataFrames, statement: str) -> DataFrame:
    conn = duckdb.connect()
    try:
        for k, v in dfs.items():
            duckdb.from_arrow(v.as_arrow(), connection=conn).create_view(k)
        return ArrowDataFrame(_duck_as_arrow(conn.execute(statement)))
    finally:
        conn.close()