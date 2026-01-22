from __future__ import annotations
from typing import Optional
import torch
from . import _binary_ufuncs_impl, _dtypes_impl, _unary_ufuncs_impl, _util
from ._normalizations import (
def _ufunc_postprocess(result, out, casting):
    if out is not None:
        result = _util.typecast_tensor(result, out.dtype.torch_dtype, casting)
        result = torch.broadcast_to(result, out.shape)
    return result