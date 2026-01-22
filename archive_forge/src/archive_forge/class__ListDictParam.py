import inspect
from typing import (
import pandas as pd
import pyarrow as pa
from triad import Schema, assert_or_throw
from triad.collections.function_wrapper import (
from triad.utils.iter import EmptyAwareIterable, make_empty_aware
from ..constants import FUGUE_ENTRYPOINT
from ..dataset.api import count as df_count
from .array_dataframe import ArrayDataFrame
from .arrow_dataframe import ArrowDataFrame
from .dataframe import AnyDataFrame, DataFrame, LocalDataFrame, as_fugue_df
from .dataframe_iterable_dataframe import (
from .dataframes import DataFrames
from .iterable_dataframe import IterableDataFrame
from .pandas_dataframe import PandasDataFrame
@fugue_annotated_param(List[Dict[str, Any]])
class _ListDictParam(_LocalNoSchemaDataFrameParam):

    @no_type_check
    def to_input_data(self, df: DataFrame, ctx: Any) -> List[Dict[str, Any]]:
        return df.as_local().as_dicts()

    @no_type_check
    def to_output_df(self, output: List[Dict[str, Any]], schema: Any, ctx: Any) -> DataFrame:
        schema = schema if isinstance(schema, Schema) else Schema(schema)

        def get_all() -> Iterable[List[Any]]:
            for row in output:
                yield [row[x] for x in schema.names]
        return IterableDataFrame(get_all(), schema)

    @no_type_check
    def count(self, df: List[Dict[str, Any]]) -> int:
        return len(df)