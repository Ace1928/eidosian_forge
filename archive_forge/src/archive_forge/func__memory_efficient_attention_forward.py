from typing import Any, Optional, Sequence, Tuple, Type, Union
import torch
from . import (
from .attn_bias import (
from .common import (
from .dispatch import _dispatch_bw, _dispatch_fw, _ensure_op_supports_or_raise
def _memory_efficient_attention_forward(inp: Inputs, op: Optional[Type[AttentionFwOpBase]]) -> torch.Tensor:
    inp.validate_inputs()
    output_shape = inp.normalize_bmhk()
    if op is None:
        op = _dispatch_fw(inp, False)
    else:
        _ensure_op_supports_or_raise(ValueError, 'memory_efficient_attention', op, inp)
    out, *_ = op.apply(inp, needs_gradient=False)
    return out.reshape(output_shape)