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
def to_pandas_types_mapper(pa_type: pa.DataType, use_extension_types: bool=False, use_arrow_dtype: bool=False) -> Optional[pd.api.extensions.ExtensionDtype]:
    """The types_mapper for ``pa.Table.to_pandas``

    :param pa_type: the pyarrow data type
    :param use_extension_types: whether to use pandas extension
        data types, default to False
    :param use_arrow_dtype: if True and when pandas supports ``ArrowDType``,
        use pyarrow types, default False
    :return: the pandas ExtensionDtype if available, otherwise None

    .. note::

        * If ``use_extension_types`` is False and ``use_arrow_dtype`` is True,
            it converts the type to ``ArrowDType``
        * If both are true, it converts the type to the numpy backend nullable
            dtypes if possible, otherwise, it converts to ``ArrowDType``
    """
    use_arrow_dtype = use_arrow_dtype and hasattr(pd, 'ArrowDtype')
    if use_extension_types:
        return _PA_TO_PANDAS_EXTENSION_TYPE_MAP[pa_type] if pa_type in _PA_TO_PANDAS_EXTENSION_TYPE_MAP else None if not use_arrow_dtype else pd.ArrowDtype(pa_type)
    if use_arrow_dtype:
        return pd.ArrowDtype(pa_type)
    return None