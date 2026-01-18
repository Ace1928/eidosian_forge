import json
import pickle
from datetime import date, datetime
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import io
import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version
from pandas.core.dtypes.base import ExtensionDtype
from pyarrow.compute import CastOptions, binary_join_element_wise
from pyarrow.json import read_json, ParseOptions as JsonParseOptions
from triad.constants import TRIAD_VAR_QUOTE
from .convert import as_type
from .iter import EmptyAwareIterable, Slicer
from .json import loads_no_dup
from .schema import move_to_unquoted, quote_name, unquote_name
from .assertion import assert_or_throw
def to_pa_datatype(obj: Any) -> pa.DataType:
    """Convert an object to pyarrow DataType

    :param obj: any object
    :raises TypeError: if unable to convert
    :return: an instance of pd.DataType
    """
    if obj is None:
        raise TypeError("obj can't be None")
    if isinstance(obj, pa.DataType):
        return obj
    if obj is bool:
        return pa.bool_()
    if obj is int:
        return pa.int64()
    if obj is float:
        return pa.float64()
    if obj is str:
        return pa.string()
    if isinstance(obj, str):
        return _parse_type(obj)
    if isinstance(obj, ExtensionDtype):
        if obj in _PANDAS_EXTENSION_TYPE_TO_PA_MAP:
            return _PANDAS_EXTENSION_TYPE_TO_PA_MAP[obj]
        if hasattr(pd, 'ArrowDtype'):
            if isinstance(obj, pd.ArrowDtype):
                return obj.pyarrow_dtype
            if obj == pd.StringDtype('pyarrow'):
                return pa.string()
    if type(obj) == type and issubclass(obj, datetime):
        return TRIAD_DEFAULT_TIMESTAMP
    if type(obj) == type and issubclass(obj, date):
        return pa.date32()
    return pa.from_numpy_dtype(np.dtype(obj))