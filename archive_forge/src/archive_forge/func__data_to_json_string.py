import json
import os
import random
import hashlib
import warnings
from typing import Union, MutableMapping, Optional, Dict, Sequence, TYPE_CHECKING, List
import pandas as pd
from toolz import curried
from typing import TypeVar
from ._importers import import_pyarrow_interchange
from .core import sanitize_dataframe, sanitize_arrow_table, DataFrameLike
from .core import sanitize_geo_interface
from .deprecation import AltairDeprecationWarning
from .plugin_registry import PluginRegistry
from typing import Protocol, TypedDict, Literal
def _data_to_json_string(data: DataType) -> str:
    """Return a JSON string representation of the input data"""
    check_data_type(data)
    if hasattr(data, '__geo_interface__'):
        if isinstance(data, pd.DataFrame):
            data = sanitize_dataframe(data)
        data = sanitize_geo_interface(data.__geo_interface__)
        return json.dumps(data)
    elif isinstance(data, pd.DataFrame):
        data = sanitize_dataframe(data)
        return data.to_json(orient='records', double_precision=15)
    elif isinstance(data, dict):
        if 'values' not in data:
            raise KeyError('values expected in data dict, but not present.')
        return json.dumps(data['values'], sort_keys=True)
    elif isinstance(data, DataFrameLike):
        pa_table = arrow_table_from_dfi_dataframe(data)
        return json.dumps(pa_table.to_pylist())
    else:
        raise NotImplementedError('to_json only works with data expressed as a DataFrame or as a dict')