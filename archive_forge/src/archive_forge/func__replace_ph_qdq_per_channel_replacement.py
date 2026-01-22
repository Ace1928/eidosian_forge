import torch
from torch.fx import GraphModule
from ..utils import (
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.subgraph_rewriter import replace_pattern
from torch._higher_order_ops.out_dtype import out_dtype
from typing import Optional, Callable, Tuple, Any
from dataclasses import dataclass
from functools import partial
def _replace_ph_qdq_per_channel_replacement(gm: torch.fx.GraphModule):
    return _replace_literals_with_existing_placeholders(gm, exclude_literals=[-1], literal_to_ph_idx={1: 3, -128: 4, 127: 5})