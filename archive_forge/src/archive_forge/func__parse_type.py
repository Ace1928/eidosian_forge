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
def _parse_type(expr: str) -> pa.DataType:
    name, args = _parse_type_function(expr)
    if name in _TYPE_EXPRESSION_MAPPING:
        assert len(args) == 0, f"{expr} can't have arguments"
        return _TYPE_EXPRESSION_MAPPING[name]
    if name == 'decimal':
        assert 1 <= len(args) <= 2, f'{expr}, decimal must have 1 or 2 argument'
        return pa.decimal128(int(args[0]), 0 if len(args) == 1 else int(args[1]))
    if name == 'timestamp':
        assert 1 <= len(args) <= 2, f'{expr}, timestamp must have 1 or 2 arguments'
        return pa.timestamp(args[0], None if len(args) == 1 else args[1])
    raise SyntaxError(f'{expr} is not a supported type')