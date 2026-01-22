from typing import Any, Dict, Iterable, List, Optional, Tuple
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from triad import assert_or_throw
from triad.collections.schema import Schema
from triad.utils.assertion import assert_arg_not_none
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import cast_pa_table
from fugue.dataframe import DataFrame, LocalBoundedDataFrame, PandasDataFrame
from fugue.dataframe.dataframe import _input_schema
from fugue.dataframe.pandas_dataframe import _pd_as_dicts
from fugue.exceptions import FugueDataFrameOperationError
from fugue.plugins import (
from ._constants import FUGUE_DASK_USE_ARROW
from ._utils import DASK_UTILS, collect, get_default_partitions
@is_bounded.candidate(lambda df: isinstance(df, dd.DataFrame))
def _dd_is_bounded(df: dd.DataFrame) -> bool:
    return True