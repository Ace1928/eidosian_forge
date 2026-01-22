from dataclasses import replace
from enum import Enum
from functools import partial
from typing import Any, List, Mapping, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from . import attn_bias
from .attn_bias import (
from .common import (
@register_operator
class BwOp(AttentionBwOpBase):
    __doc__ = FwOp.__doc__
    OPERATOR = get_xformers_operator('efficient_attention_backward_ck')
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), torch.Tensor, LowerTriangularMask, BlockDiagonalMask, BlockDiagonalCausalMask, attn_bias.BlockDiagonalCausalFromBottomRightMask}
    SUPPORTS_ATTN_BIAS_GRAD = True
    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = FwOp.SUPPORTS_DIFFERENT_VALUE_EMBED
    NAME = 'ckB'
    _TEST_K: List[int] = [32, 128, 256]

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(BwOp, cls).not_supported_reasons(d)
        matmul_alignment_mn = _minimum_gemm_alignment(d)
        check_lastdim_alignment_stride1(reasons, 'query', d.query, matmul_alignment_mn)
        check_lastdim_alignment_stride1(reasons, 'key', d.key, matmul_alignment_mn)
        check_lastdim_alignment_stride1(reasons, 'value', d.value, matmul_alignment_mn)
        _check_bias_alignment(reasons, d.attn_bias)
        attn_bias_tensor = _get_tensor_bias(d.attn_bias)
        if attn_bias_tensor is not None and attn_bias_tensor.requires_grad:
            if d.query.ndim == 3 and attn_bias_tensor.ndim == 3:
                expected_bias_shape = (*d.query.shape[:2], d.key.shape[1])
            else:
                expected_bias_shape = (d.query.shape[0], d.query.shape[2] if d.query.ndim == 4 else 1, d.query.shape[1], d.key.shape[1])
            if tuple(attn_bias_tensor.shape) != expected_bias_shape:
                reasons.append(f'Broadcasting the `attn_bias` tensor is not supported (shape: {tuple(attn_bias_tensor.shape)}/ expected: {expected_bias_shape})')
        _check_large_shapes(reasons, d)
        reasons.append('Backward is currently not supported by ck-tiled!')
        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        if type(inp.attn_bias) not in BwOp.SUPPORTED_ATTN_BIAS_TYPES:
            raise NotImplementedError('Unsupported attn_bias type')
        seqstart_k, seqstart_q, max_seqlen_q = _get_seqlen_info(inp)
        dtype = inp.query.dtype
        rng_seed = rng_offset = 0
        if inp.p != 0.0:
            if ctx.rng_state is None or ctx.rng_state.dtype != torch.int64 or ctx.rng_state.device.type != 'cpu' or (ctx.rng_state.shape != (2,)):
                raise NotImplementedError(f'Invalid rng_state: {ctx.rng_state}')
            rng_seed, rng_offset = ctx.rng_state.tolist()
        grad_q, grad_k, grad_v, grad_bias = cls.OPERATOR(grad.to(dtype), inp.query, inp.key, inp.value, attn_bias=_get_tensor_bias(inp.attn_bias), seqstart_q=seqstart_q, seqstart_k=seqstart_k, max_seqlen_q=max_seqlen_q, seqlen_k=inp.attn_bias.k_seqinfo.seqlen if isinstance(inp.attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask) else None, logsumexp=ctx.lse, output=ctx.out.to(dtype), dropout_p=inp.p, rng_seed=rng_seed, rng_offset=rng_offset, custom_mask_type=_custom_mask_type(inp.attn_bias), scale=inp.scale)
        if not (isinstance(inp.attn_bias, torch.Tensor) and inp.attn_bias.requires_grad):
            grad_bias = None
        return Gradients(dq=grad_q, dk=grad_k, dv=grad_v, db=grad_bias)

    @classmethod
    def operator_flop(cls, dO, q, k, v, b, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, logsumexp, output, dropout_p, rng_seed, rng_offset, custom_mask_type, scale) -> int:
        return cls.attn_operator_flop(q, k, v, seqstart_q=cu_seqlens_q, seqstart_k=cu_seqlens_k, causal=custom_mask_type > 0)