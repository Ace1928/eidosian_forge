from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
def _get_column_names_and_types(self, dtype_mapper):
    column_names_and_types = []
    if self.index is not None:
        for i, idx_label in enumerate(self.index):
            idx_type = dtype_mapper(self.frame.index._get_level_values(i))
            column_names_and_types.append((str(idx_label), idx_type, True))
    column_names_and_types += [(str(self.frame.columns[i]), dtype_mapper(self.frame.iloc[:, i]), False) for i in range(len(self.frame.columns))]
    return column_names_and_types