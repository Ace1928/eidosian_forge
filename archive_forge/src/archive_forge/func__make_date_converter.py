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
def _make_date_converter(date_parser=lib.no_default, dayfirst: bool=False, cache_dates: bool=True, date_format: dict[Hashable, str] | str | None=None):
    if date_parser is not lib.no_default:
        warnings.warn("The argument 'date_parser' is deprecated and will be removed in a future version. Please use 'date_format' instead, or read your data in as 'object' dtype and then call 'to_datetime'.", FutureWarning, stacklevel=find_stack_level())
    if date_parser is not lib.no_default and date_format is not None:
        raise TypeError("Cannot use both 'date_parser' and 'date_format'")

    def unpack_if_single_element(arg):
        if isinstance(arg, np.ndarray) and arg.ndim == 1 and (len(arg) == 1):
            return arg[0]
        return arg

    def converter(*date_cols, col: Hashable):
        if len(date_cols) == 1 and date_cols[0].dtype.kind in 'Mm':
            return date_cols[0]
        if date_parser is lib.no_default:
            strs = parsing.concat_date_cols(date_cols)
            date_fmt = date_format.get(col) if isinstance(date_format, dict) else date_format
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '.*parsing datetimes with mixed time zones will raise an error', category=FutureWarning)
                str_objs = ensure_object(strs)
                try:
                    result = tools.to_datetime(str_objs, format=date_fmt, utc=False, dayfirst=dayfirst, cache=cache_dates)
                except (ValueError, TypeError):
                    return str_objs
            if isinstance(result, DatetimeIndex):
                arr = result.to_numpy()
                arr.flags.writeable = True
                return arr
            return result._values
        else:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', '.*parsing datetimes with mixed time zones will raise an error', category=FutureWarning)
                    pre_parsed = date_parser(*(unpack_if_single_element(arg) for arg in date_cols))
                    try:
                        result = tools.to_datetime(pre_parsed, cache=cache_dates)
                    except (ValueError, TypeError):
                        result = pre_parsed
                if isinstance(result, datetime.datetime):
                    raise Exception('scalar parser')
                return result
            except Exception:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', '.*parsing datetimes with mixed time zones will raise an error', category=FutureWarning)
                    pre_parsed = parsing.try_parse_dates(parsing.concat_date_cols(date_cols), parser=date_parser)
                    try:
                        return tools.to_datetime(pre_parsed)
                    except (ValueError, TypeError):
                        return pre_parsed
    return converter