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
def _batchnorm_helper(g: jit_utils.GraphContext, input, weight, bias, running_mean, running_var):
    from torch.onnx.symbolic_opset9 import _var_mean
    batch_size = _get_tensor_dim_size(input, 0)
    channel_size = _get_tensor_dim_size(input, 1)
    if weight is None or _is_none(weight):
        if channel_size is None:
            raise errors.SymbolicValueError('Unsupported: ONNX export of batch_norm for unknown channel size.', input)
        weight_value = torch.tensor([1.0] * channel_size, dtype=_type_utils.JitScalarType.from_value(input).dtype())
        weight = g.op('Constant', value_t=weight_value)
    if bias is None or _is_none(bias):
        if channel_size is None:
            raise errors.SymbolicValueError('Unsupported: ONNX export of batch_norm for unknown channel size.', input)
        bias_value = torch.tensor([0.0] * channel_size, dtype=_type_utils.JitScalarType.from_value(input).dtype())
        bias = g.op('Constant', value_t=bias_value)
    if running_mean is None or _is_none(running_mean) or running_var is None or _is_none(running_var):
        assert batch_size is not None and channel_size is not None
        reshape_in = _reshape_helper(g, input, g.op('Constant', value_t=torch.tensor([batch_size, channel_size, -1], dtype=torch.int64)))
        trans_in = g.op('Transpose', reshape_in, perm_i=[0, 2, 1])
        running_var, running_mean = _var_mean(g, trans_in, g.op('Constant', value_t=torch.tensor([0, 1], dtype=torch.int64)), False, False)
    return (weight, bias, running_mean, running_var)