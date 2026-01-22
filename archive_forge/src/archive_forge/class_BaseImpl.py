from __future__ import annotations
import io
import json
import os
from typing import (
import warnings
from warnings import catch_warnings
from pandas._config import using_pyarrow_string_dtype
from pandas._config.config import _get_option
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
import pandas as pd
from pandas import (
from pandas.core.shared_docs import _shared_docs
from pandas.io._util import arrow_string_types_mapper
from pandas.io.common import (
class BaseImpl:

    @staticmethod
    def validate_dataframe(df: DataFrame) -> None:
        if not isinstance(df, DataFrame):
            raise ValueError('to_parquet only supports IO with DataFrames')

    def write(self, df: DataFrame, path, compression, **kwargs):
        raise AbstractMethodError(self)

    def read(self, path, columns=None, **kwargs) -> DataFrame:
        raise AbstractMethodError(self)