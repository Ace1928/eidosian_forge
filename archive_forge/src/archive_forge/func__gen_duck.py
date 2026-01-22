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
def _gen_duck() -> DuckDataFrame:
    if isinstance(df, DuckDBPyRelation):
        assert_or_throw(schema is None, ValueError('schema must be None when df is a DuckDBPyRelation'))
        return DuckDataFrame(df)
    if isinstance(df, DataFrame):
        assert_or_throw(schema is None, ValueError('schema must be None when df is a DataFrame'))
        if isinstance(df, DuckDataFrame):
            return df
        rdf = DuckDataFrame(duckdb.from_arrow(df.as_arrow(), connection=engine.connection))
        rdf.reset_metadata(df.metadata if df.has_metadata else None)
        return rdf
    tdf = ArrowDataFrame(df, schema)
    return DuckDataFrame(duckdb.from_arrow(tdf.native, connection=engine.connection))