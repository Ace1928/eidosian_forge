from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from typing import (
import uuid
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas._typing import (
from pandas.errors import MergeError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.frame import _merge_doc
from pandas.core.indexes.api import default_index
from pandas.core.sorting import (
def _convert_values_for_libjoin(self, values: AnyArrayLike, side: str) -> np.ndarray:
    if not Index(values).is_monotonic_increasing:
        if isna(values).any():
            raise ValueError(f'Merge keys contain null values on {side} side')
        raise ValueError(f'{side} keys must be sorted')
    if isinstance(values, ArrowExtensionArray):
        values = values._maybe_convert_datelike_array()
    if needs_i8_conversion(values.dtype):
        values = values.view('i8')
    elif isinstance(values, BaseMaskedArray):
        values = values._data
    elif isinstance(values, ExtensionArray):
        values = values.to_numpy()
    return values