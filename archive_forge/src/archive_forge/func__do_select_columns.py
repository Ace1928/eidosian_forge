from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
def _do_select_columns(self, data: DataFrame, columns: Sequence[str]) -> DataFrame:
    if not self._column_selector_set:
        column_set = set(columns)
        if len(column_set) != len(columns):
            raise ValueError('columns contains duplicate entries')
        unmatched = column_set.difference(data.columns)
        if unmatched:
            joined = ', '.join(list(unmatched))
            raise ValueError(f'The following columns were not found in the Stata data set: {joined}')
        dtyplist = []
        typlist = []
        fmtlist = []
        lbllist = []
        for col in columns:
            i = data.columns.get_loc(col)
            dtyplist.append(self._dtyplist[i])
            typlist.append(self._typlist[i])
            fmtlist.append(self._fmtlist[i])
            lbllist.append(self._lbllist[i])
        self._dtyplist = dtyplist
        self._typlist = typlist
        self._fmtlist = fmtlist
        self._lbllist = lbllist
        self._column_selector_set = True
    return data[columns]