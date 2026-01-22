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
def _validate_parse_dates_presence(self, columns: Sequence[Hashable]) -> Iterable:
    """
        Check if parse_dates are in columns.

        If user has provided names for parse_dates, check if those columns
        are available.

        Parameters
        ----------
        columns : list
            List of names of the dataframe.

        Returns
        -------
        The names of the columns which will get parsed later if a dict or list
        is given as specification.

        Raises
        ------
        ValueError
            If column to parse_date is not in dataframe.

        """
    cols_needed: Iterable
    if is_dict_like(self.parse_dates):
        cols_needed = itertools.chain(*self.parse_dates.values())
    elif is_list_like(self.parse_dates):
        cols_needed = itertools.chain.from_iterable((col if is_list_like(col) and (not isinstance(col, tuple)) else [col] for col in self.parse_dates))
    else:
        cols_needed = []
    cols_needed = list(cols_needed)
    missing_cols = ', '.join(sorted({col for col in cols_needed if isinstance(col, str) and col not in columns}))
    if missing_cols:
        raise ValueError(f"Missing column provided to 'parse_dates': '{missing_cols}'")
    return [col if isinstance(col, str) or col in columns else columns[col] for col in cols_needed]