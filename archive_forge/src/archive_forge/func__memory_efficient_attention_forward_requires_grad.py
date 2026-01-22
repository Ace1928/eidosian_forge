from typing import Any, Optional, Sequence, Tuple, Type, Union
import torch
from . import (
from .attn_bias import (
from .common import (
from .dispatch import _dispatch_bw, _dispatch_fw, _ensure_op_supports_or_raise
def _memory_efficient_attention_forward_requires_grad(inp: Inputs, op: Optional[Type[AttentionFwOpBase]]) -> Tuple[torch.Tensor, Context]:
    inp.validate_inputs()
    output_shape = inp.normalize_bmhk()
    if op is None:
        op = _dispatch_fw(inp, True)
    else:
        _ensure_op_supports_or_raise(ValueError, 'memory_efficient_attention', op, inp)
    out = op.apply(inp, needs_gradient=True)
    assert out[1] is not None
    return (out[0].reshape(output_shape), out[1])