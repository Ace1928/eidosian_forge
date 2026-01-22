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
@head.candidate(lambda df, *args, **kwargs: isinstance(df, pd.DataFrame))
def _pd_head(df: pd.DataFrame, n: int, columns: Optional[List[str]]=None, as_fugue: bool=False) -> pd.DataFrame:
    if columns is not None:
        df = df[columns]
    return _adjust_df(df.head(n), as_fugue=as_fugue)