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
def to_hdf(path_or_buf: FilePath | HDFStore, key: str, value: DataFrame | Series, mode: str='a', complevel: int | None=None, complib: str | None=None, append: bool=False, format: str | None=None, index: bool=True, min_itemsize: int | dict[str, int] | None=None, nan_rep=None, dropna: bool | None=None, data_columns: Literal[True] | list[str] | None=None, errors: str='strict', encoding: str='UTF-8') -> None:
    """store this object, close it if we opened it"""
    if append:
        f = lambda store: store.append(key, value, format=format, index=index, min_itemsize=min_itemsize, nan_rep=nan_rep, dropna=dropna, data_columns=data_columns, errors=errors, encoding=encoding)
    else:
        f = lambda store: store.put(key, value, format=format, index=index, min_itemsize=min_itemsize, nan_rep=nan_rep, data_columns=data_columns, errors=errors, encoding=encoding, dropna=dropna)
    path_or_buf = stringify_path(path_or_buf)
    if isinstance(path_or_buf, str):
        with HDFStore(path_or_buf, mode=mode, complevel=complevel, complib=complib) as store:
            f(store)
    else:
        f(path_or_buf)