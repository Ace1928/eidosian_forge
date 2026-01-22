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
@fugue_annotated_param(EmptyAwareIterable[List[Any]])
class _EmptyAwareIterableListParam(_LocalNoSchemaDataFrameParam):

    @no_type_check
    def to_input_data(self, df: DataFrame, ctx: Any) -> EmptyAwareIterable[List[Any]]:
        return make_empty_aware(df.as_array_iterable(type_safe=True))

    @no_type_check
    def to_output_df(self, output: EmptyAwareIterable[List[Any]], schema: Any, ctx: Any) -> DataFrame:
        return IterableDataFrame(output, schema)

    @no_type_check
    def count(self, df: EmptyAwareIterable[List[Any]]) -> int:
        return sum((1 for _ in df))