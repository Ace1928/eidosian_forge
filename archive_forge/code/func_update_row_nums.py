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
def update_row_nums(match):
    """
            Update the row numbers to start at 1.

            Parameters
            ----------
            match : re.Match object
                The match from the origin `re.sub` looking for row number tags.

            Returns
            -------
            str
                The updated string with new row numbers.

            Notes
            -----
            This is needed because the parser we are using does not scale well if
            the row numbers remain because empty rows are inserted for all "missing"
            rows.
            """
    b = match.group(0)
    return re.sub(b'\\d+', lambda c: str(int(c.group(0).decode('utf-8')) - _skiprows).encode('utf-8'), b)