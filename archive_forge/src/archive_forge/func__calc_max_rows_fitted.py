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
def _calc_max_rows_fitted(self) -> int | None:
    """Number of rows with data fitting the screen."""
    max_rows: int | None
    if self._is_in_terminal():
        _, height = get_terminal_size()
        if self.max_rows == 0:
            return height - self._get_number_of_auxiliary_rows()
        if self._is_screen_short(height):
            max_rows = height
        else:
            max_rows = self.max_rows
    else:
        max_rows = self.max_rows
    return self._adjust_max_rows(max_rows)