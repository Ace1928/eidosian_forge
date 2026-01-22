from __future__ import annotations
import functools
import inspect
import sys
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _type_utils, errors
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils
from torch.types import Number
@_beartype.beartype
def _topk_helper(g: jit_utils.GraphContext, input, k, dim, largest=True, sorted=False, out=None):
    if out is not None:
        _unimplemented('TopK', 'Out parameter is not supported')
    if not _is_value(k):
        k = g.op('Constant', value_t=torch.tensor([k], dtype=torch.int64))
    else:
        k = _reshape_helper(g, k, g.op('Constant', value_t=torch.tensor([1])))
        if _try_get_scalar_type(k) != _type_utils.JitScalarType.INT64:
            k = g.op('Cast', k, to_i=_C_onnx.TensorProtoDataType.INT64)
    if g.opset <= 10:
        if not largest:
            _unimplemented('TopK', 'Ascending is not supported')
        return g.op('TopK', input, k, axis_i=dim, outputs=2)
    else:
        return g.op('TopK', input, k, axis_i=dim, largest_i=largest, sorted_i=sorted, outputs=2)