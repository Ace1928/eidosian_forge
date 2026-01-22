from __future__ import annotations
import functools
import sys
import warnings
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.onnx
from torch import _C
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::max_pool1d', decorate=[_apply_params('max_pool1d', 1, return_indices=False)])
@_onnx_symbolic('aten::max_pool2d', decorate=[_apply_params('max_pool2d', 2, return_indices=False)])
@_onnx_symbolic('aten::max_pool3d', decorate=[_apply_params('max_pool3d', 3, return_indices=False)])
@_onnx_symbolic('aten::max_pool1d_with_indices', decorate=[_apply_params('max_pool1d_with_indices', 1, return_indices=True)])
@_onnx_symbolic('aten::max_pool2d_with_indices', decorate=[_apply_params('max_pool2d_with_indices', 2, return_indices=True)])
@_onnx_symbolic('aten::max_pool3d_with_indices', decorate=[_apply_params('max_pool3d_with_indices', 3, return_indices=True)])
@_beartype.beartype
def _max_pool(name: str, expand_size: int, return_indices: bool):

    @symbolic_helper.quantized_args(True, False, False, False, False, False)
    @symbolic_helper.parse_args('v', 'is', 'is', 'is', 'is', 'i')
    def symbolic_fn(g: jit_utils.GraphContext, input: _C.Value, kernel_size: Sequence[int], stride: Sequence[int], padding: Union[int, Sequence[int]], dilation: Sequence[int], ceil_mode: bool):
        kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(expand_size, kernel_size, stride, padding, dilation)
        if return_indices:
            return _aten_max_pool_with_indices_onnx(g, input, kernel_shape, strides, pads, dilations, ceil_mode, expand_size + 1, [1] * expand_size, [0] * expand_size, [2 + i for i in range(expand_size)])
        else:
            return _aten_max_pool_onnx(g, input, kernel_shape, strides, pads, dilations, ceil_mode, expand_size + 1)
    return symbolic_fn