from typing import Any, Optional, Sequence, Tuple, Type, Union
import torch
from . import (
from .attn_bias import (
from .common import (
from .dispatch import _dispatch_bw, _dispatch_fw, _ensure_op_supports_or_raise
def _memory_efficient_attention_backward(ctx: Context, inp: Inputs, grad: torch.Tensor, op: Optional[Type[AttentionBwOpBase]]) -> Gradients:
    """Warning: grad/ctx.out is potentially in BMK format"""
    inp.validate_inputs()
    if grad.ndim != inp.query.ndim or grad.ndim != ctx.out.ndim:
        raise ValueError(f'All tensors should be either in BMK (ndim=3) or BMHK (ndim=4) format. \ngrad.shape : {grad.shape} \nout.shape  : {ctx.out.shape} \nquery.shape: {inp.query.shape}')
    shape_dq, shape_dk, shape_dv = tuple((x.shape for x in (inp.query, inp.key, inp.value)))
    inp.normalize_bmhk()
    if ctx.lse.ndim != 3 or (not isinstance(inp.attn_bias, BlockDiagonalMask) and ctx.lse.shape[0] != inp.query.shape[0]) or (isinstance(inp.attn_bias, BlockDiagonalMask) and ctx.lse.shape[0] != inp.attn_bias.q_seqinfo.seqstart.shape[0] - 1) or (ctx.lse.shape[1] != inp.query.shape[2]) or (not isinstance(inp.attn_bias, BlockDiagonalMask) and ctx.lse.shape[2] < inp.query.shape[1]):
        raise ValueError(f'Input tensors have incompatible shapes.lse.shape    : {ctx.lse.shape} \nquery.shape  : {inp.query.shape}')
    grad = bmk2bmhk(grad, 1)
    ctx.out = bmk2bmhk(ctx.out, 1)
    if op is None:
        op = _dispatch_bw(inp)
    else:
        _ensure_op_supports_or_raise(ValueError, 'memory_efficient_attention_backward', op, inp)
    grads = op.apply(ctx, inp, grad)
    grads.dq = grads.dq.reshape(shape_dq)
    grads.dk = grads.dk.reshape(shape_dk)
    grads.dv = grads.dv.reshape(shape_dv)
    return grads