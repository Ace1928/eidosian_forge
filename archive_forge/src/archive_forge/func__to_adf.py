from typing import Any, Iterable, Iterator, Optional, no_type_check
import polars as pl
import pyarrow as pa
from triad import Schema, make_empty_aware
from triad.utils.pyarrow import get_alter_func
from fugue import (
from fugue.dev import LocalDataFrameParam, fugue_annotated_param
from .polars_dataframe import PolarsDataFrame
from fugue.plugins import as_fugue_dataset
def _to_adf(output: pl.DataFrame, schema: Any) -> ArrowDataFrame:
    adf = output.to_arrow()
    if schema is None:
        return ArrowDataFrame(adf)
    _schema = schema if isinstance(schema, Schema) else Schema(schema)
    f = get_alter_func(adf.schema, _schema.pa_schema, safe=False)
    return ArrowDataFrame(f(adf))