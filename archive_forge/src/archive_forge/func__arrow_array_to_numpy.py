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
def _arrow_array_to_numpy(self, pa_array: pa.Array) -> np.ndarray:
    if isinstance(pa_array, pa.ChunkedArray):
        if isinstance(pa_array.type, _ArrayXDExtensionType):
            zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
            array: List = [row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)]
        else:
            zero_copy_only = _is_zero_copy_only(pa_array.type) and all((not _is_array_with_nulls(chunk) for chunk in pa_array.chunks))
            array: List = [row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)]
    elif isinstance(pa_array.type, _ArrayXDExtensionType):
        zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
        array: List = pa_array.to_numpy(zero_copy_only=zero_copy_only)
    else:
        zero_copy_only = _is_zero_copy_only(pa_array.type) and (not _is_array_with_nulls(pa_array))
        array: List = pa_array.to_numpy(zero_copy_only=zero_copy_only).tolist()
    if len(array) > 0:
        if any((isinstance(x, np.ndarray) and (x.dtype == object or x.shape != array[0].shape) or (isinstance(x, float) and np.isnan(x)) for x in array)):
            return np.array(array, copy=False, dtype=object)
    return np.array(array, copy=False)