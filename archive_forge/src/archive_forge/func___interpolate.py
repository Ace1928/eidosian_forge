from __future__ import annotations
import functools
import sys
import warnings
from typing import Optional, Sequence
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::__interpolate')
@symbolic_helper.quantized_args(True, False, False, False, False, False, False)
@_beartype.beartype
def __interpolate(g: jit_utils.GraphContext, input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias):
    return symbolic_helper.__interpolate_helper(g, input, size, scale_factor, mode, align_corners, recompute_scale_factor)