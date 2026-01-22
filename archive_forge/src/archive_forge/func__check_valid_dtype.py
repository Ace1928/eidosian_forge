from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from ._dtypes import _all_dtypes
import cupy as np
from cupy.cuda import Device as _Device
from cupy_backends.cuda.api import runtime
def _check_valid_dtype(dtype):
    for d in (None,) + _all_dtypes:
        if dtype is d:
            return
    raise ValueError('dtype must be one of the supported dtypes')