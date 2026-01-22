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
def _index_name(self, index, index_label):
    if index is True:
        nlevels = self.frame.index.nlevels
        if index_label is not None:
            if not isinstance(index_label, list):
                index_label = [index_label]
            if len(index_label) != nlevels:
                raise ValueError(f"Length of 'index_label' should match number of levels, which is {nlevels}")
            return index_label
        if nlevels == 1 and 'index' not in self.frame.columns and (self.frame.index.name is None):
            return ['index']
        else:
            return com.fill_missing_names(self.frame.index.names)
    elif isinstance(index, str):
        return [index]
    elif isinstance(index, list):
        return index
    else:
        return None