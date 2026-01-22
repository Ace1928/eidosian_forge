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
@fugue_annotated_param(Iterable[pd.DataFrame], matcher=lambda x: x == Iterable[pd.DataFrame] or x == Iterator[pd.DataFrame])
class _IterablePandasParam(LocalDataFrameParam):

    @no_type_check
    def to_input_data(self, df: DataFrame, ctx: Any) -> Iterable[pd.DataFrame]:
        if not isinstance(df, LocalDataFrameIterableDataFrame):
            yield df.as_pandas()
        else:
            for sub in df.native:
                yield sub.as_pandas()

    @no_type_check
    def to_output_df(self, output: Iterable[pd.DataFrame], schema: Any, ctx: Any) -> DataFrame:

        def dfs():
            _schema: Optional[Schema] = None if schema is None else Schema(schema)
            has_return = False
            for df in output:
                if _schema is not None and _schema.names != list(df.columns):
                    df = df[_schema.names]
                yield PandasDataFrame(df, _schema)
                has_return = True
            if not has_return and _schema is not None:
                yield PandasDataFrame(schema=_schema)
        return IterablePandasDataFrame(dfs())

    @no_type_check
    def count(self, df: Iterable[pd.DataFrame]) -> int:
        return sum((_.shape[0] for _ in df))

    def format_hint(self) -> Optional[str]:
        return 'pandas'