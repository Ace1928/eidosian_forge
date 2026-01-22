import functools
import warnings
import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import _type_utils, errors, symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import jit_utils, registration
def _constant_fill(g: jit_utils.GraphContext, sizes, dtype: int, const_value):
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    if not scalar_type.dtype().is_floating_point:
        result = g.op('ConstantFill', sizes, dtype_i=_type_utils.JitScalarType.FLOAT.onnx_type(), input_as_shape_i=1, value_f=const_value)
        return g.op('Cast', result, to_i=scalar_type.onnx_type())
    else:
        return g.op('ConstantFill', sizes, dtype_i=scalar_type.onnx_type(), input_as_shape_i=1, value_f=const_value)