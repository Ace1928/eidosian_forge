import functools
import torch
import torch._C._onnx as _C_onnx
from torch.onnx import (
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::unsafe_chunk')
@symbolic_helper.parse_args('v', 'i', 'i', 'i')
@_beartype.beartype
def unsafe_chunk(g: jit_utils.GraphContext, self, chunks, dim, _outputs=None):
    if _outputs is None:
        return g.op('SplitToSequence', self, g.op('Constant', value_t=torch.tensor(1, dtype=torch.long)), axis_i=dim, keepdims_i=0)
    size = symbolic_helper._get_tensor_dim_size(self, dim)
    if size is None:
        return symbolic_helper._unimplemented('unsafe_chunk', 'unknown dimension size')
    split_size = (size + chunks - 1) // chunks
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    splits = g.op('Constant', value_t=torch.tensor(splits, dtype=torch.long))
    return g.op('Split', self, splits, axis_i=dim, outputs=_outputs)