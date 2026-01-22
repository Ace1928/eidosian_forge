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
@_onnx_symbolic('aten::avg_pool1d', decorate=[_apply_params('avg_pool1d', 1)])
@_onnx_symbolic('aten::avg_pool2d', decorate=[_apply_params('avg_pool2d', 2)])
@_onnx_symbolic('aten::avg_pool3d', decorate=[_apply_params('avg_pool3d', 3)])
@_beartype.beartype
def _avg_pool(name, expand_size):

    @symbolic_helper.quantized_args(True, False, False, False, False, False, False)
    @symbolic_helper.parse_args('v', 'is', 'is', 'is', 'i', 'i', 'none')
    @_beartype.beartype
    def symbolic_fn(g, input: _C.Value, kernel_size: Sequence[int], stride: Sequence[int], padding: Union[int, Sequence[int]], ceil_mode: int, count_include_pad: int, divisor_override=None):
        kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(expand_size, kernel_size, stride, padding)
        result = g.op('AveragePool', input, ceil_mode_i=ceil_mode, count_include_pad_i=count_include_pad, kernel_shape_i=kernel_shape, pads_i=pads, strides_i=strides)
        return result
    return symbolic_fn