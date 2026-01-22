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
class SQLAlchemyEngine(BaseEngine):

    def __init__(self) -> None:
        import_optional_dependency('sqlalchemy', extra='sqlalchemy is required for SQL support.')

    def insert_records(self, table: SQLTable, con, frame, name: str, index: bool | str | list[str] | None=True, schema=None, chunksize: int | None=None, method=None, **engine_kwargs) -> int | None:
        from sqlalchemy import exc
        try:
            return table.insert(chunksize=chunksize, method=method)
        except exc.StatementError as err:
            msg = '(\\(1054, "Unknown column \'inf(e0)?\' in \'field list\'"\\))(?#\n            )|inf can not be used with MySQL'
            err_text = str(err.orig)
            if re.search(msg, err_text):
                raise ValueError('inf cannot be used with MySQL') from err
            raise err