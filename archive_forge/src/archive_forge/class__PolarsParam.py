from typing import Any, Iterable, Iterator, Optional, no_type_check
import polars as pl
import pyarrow as pa
from triad import Schema, make_empty_aware
from triad.utils.pyarrow import get_alter_func
from fugue import (
from fugue.dev import LocalDataFrameParam, fugue_annotated_param
from .polars_dataframe import PolarsDataFrame
from fugue.plugins import as_fugue_dataset
@fugue_annotated_param(pl.DataFrame)
class _PolarsParam(LocalDataFrameParam):

    def to_input_data(self, df: DataFrame, ctx: Any) -> Any:
        return pl.from_arrow(df.as_arrow())

    def to_output_df(self, output: Any, schema: Any, ctx: Any) -> DataFrame:
        assert isinstance(output, pl.DataFrame)
        return _to_adf(output, schema=schema)

    def count(self, df: Any) -> int:
        return df.shape[0]

    def format_hint(self) -> Optional[str]:
        return 'pyarrow'