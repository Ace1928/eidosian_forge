import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _meta_from_cupy_array(data: DataType, field: str, handle: ctypes.c_void_p) -> None:
    data = _transform_cupy_array(data)
    interface = bytes(json.dumps([data.__cuda_array_interface__], indent=2), 'utf-8')
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle, c_str(field), interface))