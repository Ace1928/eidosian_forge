from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
def append_to_multiple(self, d: dict, value, selector, data_columns=None, axes=None, dropna: bool=False, **kwargs) -> None:
    """
        Append to multiple tables

        Parameters
        ----------
        d : a dict of table_name to table_columns, None is acceptable as the
            values of one node (this will get all the remaining columns)
        value : a pandas object
        selector : a string that designates the indexable table; all of its
            columns will be designed as data_columns, unless data_columns is
            passed, in which case these are used
        data_columns : list of columns to create as data columns, or True to
            use all columns
        dropna : if evaluates to True, drop rows from all tables if any single
                 row in each table has all NaN. Default False.

        Notes
        -----
        axes parameter is currently not accepted

        """
    if axes is not None:
        raise TypeError('axes is currently not accepted as a parameter to append_to_multiple; you can create the tables independently instead')
    if not isinstance(d, dict):
        raise ValueError('append_to_multiple must have a dictionary specified as the way to split the value')
    if selector not in d:
        raise ValueError('append_to_multiple requires a selector that is in passed dict')
    axis = next(iter(set(range(value.ndim)) - set(_AXES_MAP[type(value)])))
    remain_key = None
    remain_values: list = []
    for k, v in d.items():
        if v is None:
            if remain_key is not None:
                raise ValueError('append_to_multiple can only have one value in d that is None')
            remain_key = k
        else:
            remain_values.extend(v)
    if remain_key is not None:
        ordered = value.axes[axis]
        ordd = ordered.difference(Index(remain_values))
        ordd = sorted(ordered.get_indexer(ordd))
        d[remain_key] = ordered.take(ordd)
    if data_columns is None:
        data_columns = d[selector]
    if dropna:
        idxs = (value[cols].dropna(how='all').index for cols in d.values())
        valid_index = next(idxs)
        for index in idxs:
            valid_index = valid_index.intersection(index)
        value = value.loc[valid_index]
    min_itemsize = kwargs.pop('min_itemsize', None)
    for k, v in d.items():
        dc = data_columns if k == selector else None
        val = value.reindex(v, axis=axis)
        filtered = {key: value for key, value in min_itemsize.items() if key in v} if min_itemsize is not None else None
        self.append(k, val, data_columns=dc, min_itemsize=filtered, **kwargs)