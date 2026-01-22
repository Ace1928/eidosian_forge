from __future__ import annotations
import builtins
import functools
import math
import sys
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C
from torch.onnx import _constants, _deprecation, _type_utils, errors, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.types import Number
@symbolic_helper.parse_args('v', 'is', 'i', 'i')
@_beartype.beartype
def _var_mean(g: jit_utils.GraphContext, input, dim, correction, keepdim):
    if dim is None:
        mean = g.op('ReduceMean', input, keepdims_i=0)
        t_mean = mean
        num_elements = numel(g, input)
    else:
        mean = g.op('ReduceMean', input, axes_i=dim, keepdims_i=keepdim)
        t_mean = g.op('ReduceMean', input, axes_i=dim, keepdims_i=1)
        redudced_dims = g.op('Shape', input)
        redudced_dims = g.op('Gather', redudced_dims, g.op('Constant', value_t=torch.tensor(dim)), axis_i=0)
        num_elements = g.op('ReduceProd', redudced_dims, keepdims_i=0)
    sub_v = g.op('Sub', input, t_mean)
    sqr_sub = g.op('Mul', sub_v, sub_v)
    keepdim_mean = 0 if dim is None else keepdim
    var = g.op('ReduceMean', sqr_sub, axes_i=dim, keepdims_i=keepdim_mean)
    if correction is None:
        correction = 1
    if correction != 0:
        num_elements = g.op('Cast', num_elements, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        one = g.op('Constant', value_t=torch.tensor(correction, dtype=torch.float))
        mul = g.op('Mul', var, num_elements)
        var = g.op('Div', mul, g.op('Sub', num_elements, one))
    return (var, mean)