from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import polars as pl
import pyarrow as pa
from triad.collections.schema import Schema
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
from fugue import ArrowDataFrame
from fugue.api import (
from fugue.dataframe.dataframe import DataFrame, LocalBoundedDataFrame, _input_schema
from fugue.dataframe.utils import (
from fugue.dataset.api import (
from fugue.exceptions import FugueDataFrameOperationError
from ._utils import build_empty_pl
@as_dicts.candidate(lambda df, *args, **kwargs: isinstance(df, pl.DataFrame))
def _pl_as_dicts(df: pl.DataFrame, columns: Optional[List[str]]=None) -> List[Dict[str, Any]]:
    _df = df if columns is None else _select_pa_columns(df, columns)
    return pa_table_as_dicts(_df.to_arrow(), columns=columns)