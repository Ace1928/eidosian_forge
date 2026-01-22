from __future__ import annotations
from collections.abc import (
from contextlib import contextmanager
from csv import QUOTE_NONE
from decimal import Decimal
from functools import partial
from io import StringIO
import math
import re
from shutil import get_terminal_size
from typing import (
import numpy as np
from pandas._config.config import (
from pandas._libs import lib
from pandas._libs.missing import NA
from pandas._libs.tslibs import (
from pandas._libs.tslibs.nattype import NaTType
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.indexes.api import (
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.reshape.concat import concat
from pandas.io.common import (
from pandas.io.formats import printing
def _get_formatted_column_labels(self, frame: DataFrame) -> list[list[str]]:
    from pandas.core.indexes.multi import sparsify_labels
    columns = frame.columns
    if isinstance(columns, MultiIndex):
        fmt_columns = columns._format_multi(sparsify=False, include_names=False)
        fmt_columns = list(zip(*fmt_columns))
        dtypes = self.frame.dtypes._values
        restrict_formatting = any((level.is_floating for level in columns.levels))
        need_leadsp = dict(zip(fmt_columns, map(is_numeric_dtype, dtypes)))

        def space_format(x, y):
            if y not in self.formatters and need_leadsp[x] and (not restrict_formatting):
                return ' ' + y
            return y
        str_columns_tuple = list(zip(*([space_format(x, y) for y in x] for x in fmt_columns)))
        if self.sparsify and len(str_columns_tuple):
            str_columns_tuple = sparsify_labels(str_columns_tuple)
        str_columns = [list(x) for x in zip(*str_columns_tuple)]
    else:
        fmt_columns = columns._format_flat(include_name=False)
        dtypes = self.frame.dtypes
        need_leadsp = dict(zip(fmt_columns, map(is_numeric_dtype, dtypes)))
        str_columns = [[' ' + x if not self._get_formatter(i) and need_leadsp[x] else x] for i, x in enumerate(fmt_columns)]
    return str_columns