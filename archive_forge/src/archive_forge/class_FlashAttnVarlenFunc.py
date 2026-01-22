from typing import Optional, Union
import torch
import torch.nn as nn
import flash_attn_2_cuda as flash_attn_cuda
class FlashAttnVarlenFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal=causal, window_size=window_size, alibi_slopes=alibi_slopes, return_softmax=return_softmax and dropout_p > 0)
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq, dk, dv = (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))
        _flash_attn_varlen_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k, ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.dropout_p, ctx.softmax_scale, ctx.causal, ctx.window_size, ctx.alibi_slopes, ctx.deterministic, rng_state=rng_state)
        dq = dq[..., :dout.shape[-1]]
        dk = dk[..., :dout.shape[-1]]
        dv = dv[..., :dout.shape[-1]]
        return (dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None)