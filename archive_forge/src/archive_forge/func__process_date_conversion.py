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
def _process_date_conversion(data_dict, converter: Callable, parse_spec, index_col, index_names, columns, keep_date_col: bool=False, dtype_backend=lib.no_default):

    def _isindex(colspec):
        return isinstance(index_col, list) and colspec in index_col or (isinstance(index_names, list) and colspec in index_names)
    new_cols = []
    new_data = {}
    orig_names = columns
    columns = list(columns)
    date_cols = set()
    if parse_spec is None or isinstance(parse_spec, bool):
        return (data_dict, columns)
    if isinstance(parse_spec, list):
        for colspec in parse_spec:
            if is_scalar(colspec) or isinstance(colspec, tuple):
                if isinstance(colspec, int) and colspec not in data_dict:
                    colspec = orig_names[colspec]
                if _isindex(colspec):
                    continue
                elif dtype_backend == 'pyarrow':
                    import pyarrow as pa
                    dtype = data_dict[colspec].dtype
                    if isinstance(dtype, ArrowDtype) and (pa.types.is_timestamp(dtype.pyarrow_dtype) or pa.types.is_date(dtype.pyarrow_dtype)):
                        continue
                data_dict[colspec] = converter(np.asarray(data_dict[colspec]), col=colspec)
            else:
                new_name, col, old_names = _try_convert_dates(converter, colspec, data_dict, orig_names)
                if new_name in data_dict:
                    raise ValueError(f'New date column already in dict {new_name}')
                new_data[new_name] = col
                new_cols.append(new_name)
                date_cols.update(old_names)
    elif isinstance(parse_spec, dict):
        for new_name, colspec in parse_spec.items():
            if new_name in data_dict:
                raise ValueError(f'Date column {new_name} already in dict')
            _, col, old_names = _try_convert_dates(converter, colspec, data_dict, orig_names, target_name=new_name)
            new_data[new_name] = col
            if len(colspec) == 1:
                new_data[colspec[0]] = col
            new_cols.append(new_name)
            date_cols.update(old_names)
    if isinstance(data_dict, DataFrame):
        data_dict = concat([DataFrame(new_data), data_dict], axis=1, copy=False)
    else:
        data_dict.update(new_data)
    new_cols.extend(columns)
    if not keep_date_col:
        for c in list(date_cols):
            data_dict.pop(c)
            new_cols.remove(c)
    return (data_dict, new_cols)