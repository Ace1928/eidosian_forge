from typing import Any, Dict, Iterable, List, Optional
from fugue.dataframe.dataframe import (
from fugue.exceptions import FugueDataFrameOperationError
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import apply_schema
@as_fugue_dataset.candidate(lambda df, **kwargs: isinstance(df, list), priority=0.9)
def _arr_to_fugue(df: List[Any], **kwargs: Any) -> ArrayDataFrame:
    return ArrayDataFrame(df, **kwargs)