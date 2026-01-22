import functools
import inspect
import os
from csv import Dialect
from typing import Callable, Dict, Sequence, Tuple, Union
import pandas
import pandas._libs.lib as lib
import pyarrow as pa
from pandas.core.dtypes.common import is_list_like
from pandas.io.common import get_handle, is_url
from pyarrow.csv import ConvertOptions, ParseOptions, ReadOptions, read_csv
from modin.core.io import BaseIO
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher
from modin.error_message import ErrorMessage
from modin.experimental.core.execution.native.implementations.hdk_on_native.dataframe.dataframe import (
from modin.experimental.core.storage_formats.hdk.query_compiler import (
from modin.utils import _inherit_docstrings
class ArrowEngineException(Exception):
    """Exception raised in case of Arrow engine-specific incompatibilities are found."""