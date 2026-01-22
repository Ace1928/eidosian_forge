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
@_beartype.beartype
def _op_with_optional_float_cast(g: jit_utils.GraphContext, op_name, *args, **kwargs):
    """Some PyTorch operators (e.g., Clip/Min/ReLU/Pad) are super set of ONNX in terms of data types.
    This function maximizes the exportability of PyTorch-ONNX by allowing ONNX-unsupported PyTorch
    operator data type. For example, `Cast<int>(Clip<float>(Cast<float>(INPUT)))` can be used to mimic
    `Clip<int>(INPUT)` (opset version < 12).

    Args:
        g (torch._C.Graph): graph to write the ONNX representation into.
        op_name (str): operator name in ONNX.
        *args (tuple): operands to the operator.
        **kwargs (dict): attributes to the operator along with "opset_before" (optional, None by default)
            indicating the smallest opset version to trigger such casting behavior and "target_float_t"
            (optional, torch.onnx.JitScalarType.FLOAT by default) indicating the data type of internal operator.

    Returns:
        Optional[torch._C.Value, Tuple[torch._C.Value, ...]]: output(s) of the operator.
    """
    opset_before = kwargs.pop('opset_before', None)
    target_float_t = kwargs.pop('target_float_t', _type_utils.JitScalarType.FLOAT)
    inputs = list(args)
    dtype_0 = _type_utils.JitScalarType.from_value(inputs[0])
    require_cast = not symbolic_helper._is_fp(inputs[0]) and (opset_before is None or GLOBALS.export_onnx_opset_version < opset_before)
    if require_cast:
        for input in inputs:
            if input.isCompleteTensor():
                input_scalar_type = _type_utils.JitScalarType.from_value(input)
                if input_scalar_type != dtype_0:
                    raise errors.SymbolicValueError(f'Inputs of {op_name} must have same dtype.Got {dtype_0.scalar_name()} and {input_scalar_type.scalar_name()}', input)
        for i, input in enumerate(inputs):
            if input.isCompleteTensor() and (not symbolic_helper._is_fp(input)):
                inputs[i] = g.op('Cast', input, to_i=target_float_t.onnx_type())
    self = g.op(op_name, *inputs, **kwargs)
    if require_cast:
        self = g.op('Cast', self, to_i=dtype_0.onnx_type())
    return self