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
def _filter_usecols(self, names: Sequence[Hashable]) -> Sequence[Hashable]:
    usecols = self._evaluate_usecols(self.usecols, names)
    if usecols is not None and len(names) != len(usecols):
        names = [name for i, name in enumerate(names) if i in usecols or name in usecols]
    return names