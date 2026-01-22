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
def __interpolate_helper(g: jit_utils.GraphContext, input, size, scale_factor, mode, align_corners, recompute_scale_factor):
    mode = _maybe_get_const(mode, 's')
    if 'linear' in mode:
        mode = 'linear'
    if 'cubic' in mode:
        mode = 'cubic'
    align_corners = _maybe_get_const(align_corners, 'b')
    align_corners = False if not isinstance(align_corners, bool) else align_corners
    coordinate_transformation_mode = 'asymmetric' if mode == 'nearest' else 'align_corners' if align_corners else 'half_pixel'
    if not _is_none(size):
        input_size = g.op('Shape', input)
        input_size = _slice_helper(g, input_size, axes=[0], ends=[2], starts=[0])
        try:
            is_scalar = not _is_packed_list(size) and _maybe_get_const(size, 't').dim() == 0
        except AttributeError:
            is_scalar = not _is_packed_list(size)
            if not is_scalar:
                warnings.warn('Cannot verify if the output_size is a scalar while exporting interpolate. Assuming that it is not a scalar.')
        if is_scalar:
            rank = _get_tensor_rank(input)
            if rank is None:
                return _unimplemented('interpolate (with a scalar output_size)', 'missing input shape (try giving an array of output_size values)')
            size = _unsqueeze_helper(g, size, [0])
            size = [size for i in range(rank - 2)]
            size = g.op('Concat', *size, axis_i=0)
        size = g.op('Cast', size, to_i=_C_onnx.TensorProtoDataType.INT64)
        size = g.op('Concat', input_size, size, axis_i=0)
        if g.opset >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
            empty_scales = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op('Constant', value_t=torch.tensor([], dtype=torch.float32))
            empty_scales = g.op('Constant', value_t=torch.tensor([], dtype=torch.float32))
        return g.op('Resize', input, empty_roi, empty_scales, size, coordinate_transformation_mode_s=coordinate_transformation_mode, cubic_coeff_a_f=-0.75, mode_s=mode, nearest_mode_s='floor')
    else:
        rank = _get_tensor_rank(input)
        if rank is None:
            return _unimplemented('interpolate (with scales)', 'missing input shape')
        if g.opset >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op('Constant', value_t=torch.tensor([], dtype=torch.float32))
        scales = _interpolate_get_scales(g, scale_factor, rank)
        return g.op('Resize', input, empty_roi, scales, coordinate_transformation_mode_s=coordinate_transformation_mode, cubic_coeff_a_f=-0.75, mode_s=mode, nearest_mode_s='floor')