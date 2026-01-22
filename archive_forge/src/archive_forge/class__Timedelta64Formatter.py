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
class _Timedelta64Formatter(_GenericArrayFormatter):
    values: TimedeltaArray

    def __init__(self, values: TimedeltaArray, nat_rep: str='NaT', **kwargs) -> None:
        super().__init__(values, **kwargs)
        self.nat_rep = nat_rep

    def _format_strings(self) -> list[str]:
        formatter = self.formatter or get_format_timedelta64(self.values, nat_rep=self.nat_rep, box=False)
        return [formatter(x) for x in self.values]