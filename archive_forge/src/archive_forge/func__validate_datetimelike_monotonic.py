from __future__ import annotations
import copy
from datetime import timedelta
from functools import partial
import inspect
from textwrap import dedent
from typing import (
import numpy as np
from pandas._libs.tslibs import (
import pandas._libs.window.aggregations as window_aggregations
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DataError
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core._numba import executor
from pandas.core.algorithms import factorize
from pandas.core.apply import ResamplerWindowApply
from pandas.core.arrays import ExtensionArray
from pandas.core.base import SelectionMixin
import pandas.core.common as com
from pandas.core.indexers.objects import (
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.util.numba_ import (
from pandas.core.window.common import (
from pandas.core.window.doc import (
from pandas.core.window.numba_ import (
from pandas.core.arrays.datetimelike import dtype_to_unit
def _validate_datetimelike_monotonic(self):
    """
        Validate that each group in self._on is monotonic
        """
    if self._on.hasnans:
        self._raise_monotonic_error('values must not have NaT')
    for group_indices in self._grouper.indices.values():
        group_on = self._on.take(group_indices)
        if not (group_on.is_monotonic_increasing or group_on.is_monotonic_decreasing):
            on = 'index' if self.on is None else self.on
            raise ValueError(f'Each group within {on} must be monotonic. Sort the values in {on} first.')