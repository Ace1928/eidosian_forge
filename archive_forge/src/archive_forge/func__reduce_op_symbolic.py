import functools
import torch
import torch._C._onnx as _C_onnx
from torch.onnx import (
from torch.onnx._internal import _beartype, jit_utils, registration
@_beartype.beartype
def _reduce_op_symbolic(onnx_op_name):

    @_beartype.beartype
    def symbolic(g, self, dim=None, keepdim=None):
        self = opset9._maybe_cast_reduce_op_input(g, self)
        if dim is None:
            return symbolic_helper._handle_reduce_dim_none(g, self, onnx_op_name)
        else:
            keepdim = symbolic_helper._get_const(keepdim, 'i', 'keepdim')
            return g.op(onnx_op_name, self, dim, keepdims_i=keepdim)
    return symbolic