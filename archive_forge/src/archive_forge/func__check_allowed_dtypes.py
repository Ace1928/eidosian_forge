from __future__ import annotations
import operator
from enum import IntEnum
from ._creation_functions import asarray
from ._dtypes import (
from typing import TYPE_CHECKING, Optional, Tuple, Union, Any, SupportsIndex
import types
import cupy as np
from cupy.cuda import Device as _Device
from cupy.cuda import stream as stream_module
from cupy_backends.cuda.api import runtime
from cupy import array_api
def _check_allowed_dtypes(self, other: Union[bool, int, float, Array], dtype_category: str, op: str) -> Array:
    """
        Helper function for operators to only allow specific input dtypes

        Use like

            other = self._check_allowed_dtypes(other, 'numeric', '__add__')
            if other is NotImplemented:
                return other
        """
    if self.dtype not in _dtype_categories[dtype_category]:
        raise TypeError(f'Only {dtype_category} dtypes are allowed in {op}')
    if isinstance(other, (int, float, bool)):
        other = self._promote_scalar(other)
    elif isinstance(other, Array):
        if other.dtype not in _dtype_categories[dtype_category]:
            raise TypeError(f'Only {dtype_category} dtypes are allowed in {op}')
    else:
        return NotImplemented
    res_dtype = _result_type(self.dtype, other.dtype)
    if op.startswith('__i'):
        if res_dtype != self.dtype:
            raise TypeError(f'Cannot perform {op} with dtypes {self.dtype} and {other.dtype}')
    return other