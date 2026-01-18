from typing import Any, Dict, Iterable, List, Optional
from fugue.dataframe.array_dataframe import ArrayDataFrame
from fugue.dataframe.dataframe import (
from fugue.exceptions import FugueDataFrameOperationError
from triad.collections.schema import Schema
from triad.utils.iter import EmptyAwareIterable, make_empty_aware
from triad.utils.pyarrow import apply_schema
Iterable of native python arrays