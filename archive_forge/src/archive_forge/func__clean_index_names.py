from __future__ import annotations
from collections import defaultdict
from copy import copy
import csv
import datetime
from enum import Enum
import itertools
from typing import (
import warnings
import numpy as np
from pandas._libs import (
import pandas._libs.ops as libops
from pandas._libs.parsers import STR_NA_VALUES
from pandas._libs.tslibs import parsing
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import (
from pandas.core import algorithms
from pandas.core.arrays import (
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools
from pandas.io.common import is_potential_multi_index
@final
def _clean_index_names(self, columns, index_col) -> tuple[list | None, list, list]:
    if not is_index_col(index_col):
        return (None, columns, index_col)
    columns = list(columns)
    if not columns:
        return ([None] * len(index_col), columns, index_col)
    cp_cols = list(columns)
    index_names: list[str | int | None] = []
    index_col = list(index_col)
    for i, c in enumerate(index_col):
        if isinstance(c, str):
            index_names.append(c)
            for j, name in enumerate(cp_cols):
                if name == c:
                    index_col[i] = j
                    columns.remove(name)
                    break
        else:
            name = cp_cols[c]
            columns.remove(name)
            index_names.append(name)
    for i, name in enumerate(index_names):
        if isinstance(name, str) and name in self.unnamed_cols:
            index_names[i] = None
    return (index_names, columns, index_col)