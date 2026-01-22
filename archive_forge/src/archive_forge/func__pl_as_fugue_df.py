from typing import Any, Iterable, Iterator, Optional, no_type_check
import polars as pl
import pyarrow as pa
from triad import Schema, make_empty_aware
from triad.utils.pyarrow import get_alter_func
from fugue import (
from fugue.dev import LocalDataFrameParam, fugue_annotated_param
from .polars_dataframe import PolarsDataFrame
from fugue.plugins import as_fugue_dataset
@as_fugue_dataset.candidate(lambda df, **kwargs: isinstance(df, pl.DataFrame))
def _pl_as_fugue_df(df: pl.DataFrame, **kwargs: Any) -> PolarsDataFrame:
    return PolarsDataFrame(df, **kwargs)