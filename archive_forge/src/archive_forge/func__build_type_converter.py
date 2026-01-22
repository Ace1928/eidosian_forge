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
def _build_type_converter(self, tp: pa.DataType) -> Callable[[Any], Any]:
    if tp in _TypeConverter._CONVERTERS:
        return _TypeConverter._CONVERTERS[tp]
    elif pa.types.is_timestamp(tp):
        return _to_pydatetime
    elif pa.types.is_decimal(tp):
        raise NotImplementedError('decimal conversion is not supported')
    elif pa.types.is_struct(tp):
        if not self._deep:
            return lambda x: _assert_pytype(dict, x)
        else:
            converters = {x.name: self._build_field_converter(x) for x in list(tp)}
            return lambda x: _to_pydict(converters, x, self._str_as_json)
    elif pa.types.is_list(tp):
        if not self._deep:
            return lambda x: _assert_pytype(list, x)
        else:
            converter = self._build_type_converter(tp.value_type)
            return lambda x: _to_pylist(converter, x, self._copy, self._str_as_json)
    elif pa.types.is_map(tp):
        if not self._deep:
            return lambda x: _assert_pytype((dict, list), x)
        else:
            k = self._build_type_converter(tp.key_type)
            v = self._build_type_converter(tp.item_type)
            return lambda x: _to_pymap(k, v, x, self._str_as_json)
    raise NotImplementedError