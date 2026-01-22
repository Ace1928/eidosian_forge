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
def _initialize_colspace(self, col_space: ColspaceArgType | None) -> ColspaceType:
    result: ColspaceType
    if col_space is None:
        result = {}
    elif isinstance(col_space, (int, str)):
        result = {'': col_space}
        result.update({column: col_space for column in self.frame.columns})
    elif isinstance(col_space, Mapping):
        for column in col_space.keys():
            if column not in self.frame.columns and column != '':
                raise ValueError(f'Col_space is defined for an unknown column: {column}')
        result = col_space
    else:
        if len(self.frame.columns) != len(col_space):
            raise ValueError(f'Col_space length({len(col_space)}) should match DataFrame number of columns({len(self.frame.columns)})')
        result = dict(zip(self.frame.columns, col_space))
    return result