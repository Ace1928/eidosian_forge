from typing import Any
import dask.dataframe as dd
from dask.distributed import Client
from fugue import DataFrame
from fugue.dev import (
from fugue.plugins import (
from fugue_dask._utils import DASK_UTILS
from fugue_dask.dataframe import DaskDataFrame
from fugue_dask.execution_engine import DaskExecutionEngine
@fugue_annotated_param(dd.DataFrame)
class _DaskDataFrameParam(DataFrameParam):

    def to_input_data(self, df: DataFrame, ctx: Any) -> Any:
        assert isinstance(ctx, DaskExecutionEngine)
        return ctx.to_df(df).native

    def to_output_df(self, output: Any, schema: Any, ctx: Any) -> DataFrame:
        assert isinstance(output, dd.DataFrame)
        assert isinstance(ctx, DaskExecutionEngine)
        return ctx.to_df(output, schema=schema)

    def count(self, df: DataFrame) -> int:
        raise NotImplementedError('not allowed')