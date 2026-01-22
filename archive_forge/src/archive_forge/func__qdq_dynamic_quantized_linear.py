import torch
from torch.fx import GraphModule
from ..utils import (
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.subgraph_rewriter import replace_pattern
from torch._higher_order_ops.out_dtype import out_dtype
from typing import Optional, Callable, Tuple, Any
from dataclasses import dataclass
from functools import partial
def _qdq_dynamic_quantized_linear(x_fp32, x_quant_min, x_quant_max, x_eps, weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, bias_fp32):
    x_scale, x_zero_point = torch.ops.quantized_decomposed.choose_qparams(x_fp32, x_quant_min, x_quant_max, x_eps, torch.int8)
    x_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(x_fp32, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
    out_fp32 = torch.ops.aten.linear.default(x_fp32, weight_fp32, bias_fp32)
    return out_fp32