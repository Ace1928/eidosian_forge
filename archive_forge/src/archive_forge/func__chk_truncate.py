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
def _chk_truncate(self) -> None:
    self.tr_row_num: int | None
    min_rows = self.min_rows
    max_rows = self.max_rows
    is_truncated_vertically = max_rows and len(self.series) > max_rows
    series = self.series
    if is_truncated_vertically:
        max_rows = cast(int, max_rows)
        if min_rows:
            max_rows = min(min_rows, max_rows)
        if max_rows == 1:
            row_num = max_rows
            series = series.iloc[:max_rows]
        else:
            row_num = max_rows // 2
            series = concat((series.iloc[:row_num], series.iloc[-row_num:]))
        self.tr_row_num = row_num
    else:
        self.tr_row_num = None
    self.tr_series = series
    self.is_truncated_vertically = is_truncated_vertically