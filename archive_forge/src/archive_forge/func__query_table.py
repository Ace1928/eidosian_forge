from collections.abc import Mapping, MutableMapping
from functools import partial
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version
from .. import config
from ..features import Features
from ..features.features import _ArrayXDExtensionType, _is_zero_copy_only, decode_nested_example, pandas_types_mapper
from ..table import Table
from ..utils.py_utils import no_op_if_value_is_null
def _query_table(table: Table, key: Union[int, slice, range, str, Iterable]) -> pa.Table:
    """
    Query a pyarrow Table to extract the subtable that correspond to the given key.
    """
    if isinstance(key, int):
        return table.fast_slice(key % table.num_rows, 1)
    if isinstance(key, slice):
        key = range(*key.indices(table.num_rows))
    if isinstance(key, range):
        if _is_range_contiguous(key) and key.start >= 0:
            return table.fast_slice(key.start, key.stop - key.start)
        else:
            pass
    if isinstance(key, str):
        return table.table.drop([column for column in table.column_names if column != key])
    if isinstance(key, Iterable):
        key = np.fromiter(key, np.int64)
        if len(key) == 0:
            return table.table.slice(0, 0)
        return table.fast_gather(key % table.num_rows)
    _raise_bad_key_type(key)