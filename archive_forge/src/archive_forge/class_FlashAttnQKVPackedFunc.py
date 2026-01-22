import math
import torch
import triton
import triton.language as tl
class FlashAttnQKVPackedFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, bias=None, causal=False, softmax_scale=None):
        """
        qkv: (batch, seqlen, 3, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen, seqlen).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen, seqlen)
        """
        if qkv.stride(-1) != 1:
            qkv = qkv.contiguous()
        o, lse, ctx.softmax_scale = _flash_attn_forward(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], bias=bias, causal=causal, softmax_scale=softmax_scale)
        ctx.save_for_backward(qkv, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        qkv, o, lse, bias = ctx.saved_tensors
        assert not ctx.needs_input_grad[1], 'FlashAttention does not support bias gradient yet'
        with torch.inference_mode():
            dqkv = torch.empty_like(qkv)
            _flash_attn_backward(do, qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], o, lse, dqkv[:, :, 0], dqkv[:, :, 1], dqkv[:, :, 2], bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
        return (dqkv, None, None, None)