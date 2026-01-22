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
def _value_formatter(self, float_format: FloatFormatType | None=None, threshold: float | None=None) -> Callable:
    """Returns a function to be applied on each value to format it"""
    if float_format is None:
        float_format = self.float_format
    if float_format:

        def base_formatter(v):
            assert float_format is not None
            return float_format(value=v) if notna(v) else self.na_rep
    else:

        def base_formatter(v):
            return str(v) if notna(v) else self.na_rep
    if self.decimal != '.':

        def decimal_formatter(v):
            return base_formatter(v).replace('.', self.decimal, 1)
    else:
        decimal_formatter = base_formatter
    if threshold is None:
        return decimal_formatter

    def formatter(value):
        if notna(value):
            if abs(value) > threshold:
                return decimal_formatter(value)
            else:
                return decimal_formatter(0.0)
        else:
            return self.na_rep
    return formatter