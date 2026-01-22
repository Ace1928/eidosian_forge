from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._libs import (
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DtypeWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.concat import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.io.common import (
from pandas.io.parsers.base_parser import (
def _maybe_parse_dates(self, values, index: int, try_parse_dates: bool=True):
    if try_parse_dates and self._should_parse_dates(index):
        values = self._date_conv(values, col=self.index_names[index] if self.index_names is not None else None)
    return values