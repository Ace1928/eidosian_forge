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
@as_array.candidate(lambda df, *args, **kwargs: isinstance(df, pd.DataFrame))
def _pd_as_array(df: pd.DataFrame, columns: Optional[List[str]]=None, type_safe: bool=False) -> List[Any]:
    return list(_pd_as_array_iterable(df, columns, type_safe=type_safe))