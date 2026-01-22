from __future__ import annotations
import warnings
from types import ModuleType
from typing import Any
import numpy as np
from xarray.namedarray._typing import (
from xarray.namedarray.core import NamedArray
def _get_data_namespace(x: NamedArray[Any, Any]) -> ModuleType:
    if isinstance(x._data, _arrayapi):
        return x._data.__array_namespace__()
    return np