import contextlib
import json
import os
import warnings
from io import BytesIO, IOBase, TextIOWrapper
from typing import Any, NamedTuple
import fsspec
import numpy as np
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.concat import union_categoricals
from pandas.io.common import infer_compression
from pandas.util._decorators import doc
from modin.config import MinPartitionSize
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.db_conn import ModinDatabaseConnection
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import ModinAssumptionError
class ParquetFileToRead(NamedTuple):
    """
    Class to store path and row group information for parquet reads.

    Parameters
    ----------
    path : str, path object or file-like object
        Name of the file to read.
    row_group_start : int
        Row group to start read from.
    row_group_end : int
        Row group to stop read.
    """
    path: Any
    row_group_start: int
    row_group_end: int