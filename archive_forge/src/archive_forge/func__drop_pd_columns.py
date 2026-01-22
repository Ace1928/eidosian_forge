from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from triad import assert_or_throw
from triad.collections.schema import Schema
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import pa_batch_to_dicts
from fugue.dataset.api import (
from fugue.exceptions import FugueDataFrameOperationError
from .api import (
from .dataframe import DataFrame, LocalBoundedDataFrame, _input_schema
@drop_columns.candidate(lambda df, *args, **kwargs: isinstance(df, pd.DataFrame))
def _drop_pd_columns(df: pd.DataFrame, columns: List[str], as_fugue: bool=False) -> Any:
    cols = [x for x in df.columns if x not in columns]
    if len(cols) == 0:
        raise FugueDataFrameOperationError('cannot drop all columns')
    if len(cols) + len(columns) != len(df.columns):
        _assert_no_missing(df, columns)
    return _adjust_df(df[cols], as_fugue=as_fugue)