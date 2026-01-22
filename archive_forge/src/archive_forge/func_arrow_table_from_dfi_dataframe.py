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
def arrow_table_from_dfi_dataframe(dfi_df: DataFrameLike) -> 'pyarrow.lib.Table':
    """Convert a DataFrame Interchange Protocol compatible object to an Arrow Table"""
    import pyarrow as pa
    for convert_method_name in ('arrow', 'to_arrow', 'to_arrow_table'):
        convert_method = getattr(dfi_df, convert_method_name, None)
        if callable(convert_method):
            result = convert_method()
            if isinstance(result, pa.Table):
                return result
    pi = import_pyarrow_interchange()
    return pi.from_dataframe(dfi_df)