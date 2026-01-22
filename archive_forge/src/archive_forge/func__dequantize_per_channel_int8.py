import torch
from torch.fx import GraphModule
from ..utils import (
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.subgraph_rewriter import replace_pattern
from torch._higher_order_ops.out_dtype import out_dtype
from typing import Optional, Callable, Tuple, Any
from dataclasses import dataclass
from functools import partial
def _dequantize_per_channel_int8(x_i8, scales, zero_points, ch_axis, quant_min, quant_max):
    out_fp32 = torch.ops.quantized_decomposed.dequantize_per_channel(x_i8, scales, zero_points, ch_axis, quant_min, quant_max, torch.int8)
    return out_fp32