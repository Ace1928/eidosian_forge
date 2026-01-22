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
def _truncate_vertically(self) -> None:
    """Remove rows, which are not to be displayed.

        Attributes affected:
            - tr_frame
            - tr_row_num
        """
    assert self.max_rows_fitted is not None
    row_num = self.max_rows_fitted // 2
    if row_num >= 1:
        _len = len(self.tr_frame)
        _slice = np.hstack([np.arange(row_num), np.arange(_len - row_num, _len)])
        self.tr_frame = self.tr_frame.iloc[_slice]
    else:
        row_num = cast(int, self.max_rows)
        self.tr_frame = self.tr_frame.iloc[:row_num, :]
    self.tr_row_num = row_num