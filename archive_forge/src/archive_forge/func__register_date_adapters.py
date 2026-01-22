from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
def _register_date_adapters(self) -> None:
    import sqlite3

    def _adapt_time(t) -> str:
        return f'{t.hour:02d}:{t.minute:02d}:{t.second:02d}.{t.microsecond:06d}'
    adapt_date_iso = lambda val: val.isoformat()
    adapt_datetime_iso = lambda val: val.isoformat(' ')
    sqlite3.register_adapter(time, _adapt_time)
    sqlite3.register_adapter(date, adapt_date_iso)
    sqlite3.register_adapter(datetime, adapt_datetime_iso)
    convert_date = lambda val: date.fromisoformat(val.decode())
    convert_timestamp = lambda val: datetime.fromisoformat(val.decode())
    sqlite3.register_converter('date', convert_date)
    sqlite3.register_converter('timestamp', convert_timestamp)