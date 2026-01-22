import torch
from torch.fx import GraphModule
from ..utils import (
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.subgraph_rewriter import replace_pattern
from torch._higher_order_ops.out_dtype import out_dtype
from typing import Optional, Callable, Tuple, Any
from dataclasses import dataclass
from functools import partial
def _reference_quantized_add_relu(x_i8, x_scale, x_zero_point, y_i8, y_scale, y_zero_point, out_scale, out_zero_point, quant_min, quant_max):
    """
    See comments for `_reference_quantized_add` for more information on
    how to derive the formula for out_i8 based on x_i8 and y_i8
    """
    x_i32 = x_i8.to(torch.int32)
    y_i32 = y_i8.to(torch.int32)
    x_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, x_i32 - x_zero_point, x_scale / out_scale)
    y_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, y_i32 - y_zero_point, y_scale / out_scale)
    out_i32 = x_i32 + y_i32 + out_zero_point
    out_i8 = torch.ops.aten.clamp(out_i32, out_zero_point, quant_max).to(torch.int8)
    return out_i8