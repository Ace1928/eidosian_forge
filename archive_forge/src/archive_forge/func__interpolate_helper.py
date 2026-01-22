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
def _interpolate_helper(name, dim, interpolate_mode):

    @quantized_args(True, False, False)
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = _get_interpolate_attributes(g, interpolate_mode, args)
        align_corners = _maybe_get_scalar(align_corners)
        coordinate_transformation_mode = 'asymmetric' if interpolate_mode == 'nearest' else 'align_corners' if align_corners else 'half_pixel'
        if scales is None:
            input_size = g.op('Shape', input)
            input_size_beg = _slice_helper(g, input_size, axes=[0], ends=[2], starts=[0])
            output_size = g.op('Cast', output_size, to_i=_C_onnx.TensorProtoDataType.INT64)
            output_size = g.op('Concat', input_size_beg, output_size, axis_i=0)
            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
                empty_scales = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op('Constant', value_t=torch.tensor([], dtype=torch.float32))
                empty_scales = g.op('Constant', value_t=torch.tensor([], dtype=torch.float32))
            return g.op('Resize', input, empty_roi, empty_scales, output_size, coordinate_transformation_mode_s=coordinate_transformation_mode, cubic_coeff_a_f=-0.75, mode_s=interpolate_mode, nearest_mode_s='floor')
        else:
            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op('Constant', value_t=torch.tensor([], dtype=torch.float32))
            return g.op('Resize', input, empty_roi, scales, coordinate_transformation_mode_s=coordinate_transformation_mode, cubic_coeff_a_f=-0.75, mode_s=interpolate_mode, nearest_mode_s='floor')
    return symbolic_fn