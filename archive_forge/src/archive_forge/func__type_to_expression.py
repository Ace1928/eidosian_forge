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
def _type_to_expression(dt: pa.DataType) -> str:
    if dt in _TYPE_EXPRESSION_R_MAPPING:
        return _TYPE_EXPRESSION_R_MAPPING[dt]
    if isinstance(dt, pa.TimestampType):
        if dt.tz is None:
            return 'datetime'
        else:
            return f'timestamp({dt.unit},{dt.tz})'
    if isinstance(dt, pa.Decimal128Type):
        if dt.scale == 0:
            return f'decimal({dt.precision})'
        else:
            return f'decimal({dt.precision},{dt.scale})'
    if isinstance(dt, pa.StructType):
        f = ','.join((_field_to_expression(x) for x in list(dt)))
        return '{' + f + '}'
    if isinstance(dt, pa.ListType):
        f = _type_to_expression(dt.value_type)
        return '[' + f + ']'
    if isinstance(dt, pa.MapType):
        k = _type_to_expression(dt.key_type)
        v = _type_to_expression(dt.item_type)
        return '<' + k + ',' + v + '>'
    raise NotImplementedError(f'{dt} is not supported')