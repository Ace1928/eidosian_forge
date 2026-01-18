import codecs
import functools
import importlib
import inspect
import json
import os
import re
import sys
import types
import warnings
from pathlib import Path
from textwrap import dedent, indent
from typing import (
import numpy as np
import pandas
from packaging import version
from pandas._typing import JSONSerializable
from pandas.util._decorators import Appender  # type: ignore
from pandas.util._print_versions import (  # type: ignore[attr-defined]
from modin._version import get_versions
from modin.config import DocModule, Engine, StorageFormat
def try_cast_to_pandas(obj: Any, squeeze: bool=False) -> Any:
    """
    Convert `obj` and all nested objects from Modin to pandas if it is possible.

    If no convertion possible return `obj`.

    Parameters
    ----------
    obj : object
        Object to convert from Modin to pandas.
    squeeze : bool, default: False
        Squeeze the converted object(s) before returning them.

    Returns
    -------
    object
        Converted object.
    """
    if isinstance(obj, SupportsPublicToPandas) or hasattr(obj, 'modin'):
        result = obj.modin.to_pandas() if hasattr(obj, 'modin') else obj.to_pandas()
        if squeeze:
            result = result.squeeze(axis=1)
        if isinstance(getattr(result, 'name', None), str) and result.name == MODIN_UNNAMED_SERIES_LABEL:
            result.name = None
        return result
    if isinstance(obj, (list, tuple)):
        return type(obj)([try_cast_to_pandas(o, squeeze=squeeze) for o in obj])
    if isinstance(obj, dict):
        return {k: try_cast_to_pandas(v, squeeze=squeeze) for k, v in obj.items()}
    if callable(obj):
        module_hierarchy = getattr(obj, '__module__', '').split('.')
        fn_name = getattr(obj, '__name__', None)
        if fn_name and module_hierarchy[0] == 'modin':
            return getattr(pandas.DataFrame, fn_name, obj) if module_hierarchy[-1] == 'dataframe' else getattr(pandas.Series, fn_name, obj)
    return obj