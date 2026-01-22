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
def _assert_no_missing(df: pd.DataFrame, columns: Iterable[Any]) -> None:
    missing = [x for x in columns if x not in df.columns]
    if len(missing) > 0:
        raise FugueDataFrameOperationError('found nonexistent columns: {missing}')