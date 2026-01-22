import flash_attn_cuda
import torch
import torch.nn as nn
class FlashBlocksparseAttnFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal):
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        context, softmax_lse, S_dmask = _flash_blocksparse_attn_forward(qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal=causal, return_softmax=False)
        ctx.save_for_backward(qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_s = max_s
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return context

    @staticmethod
    def backward(ctx, dout):
        qkv, context, S_dmask, softmax_lse, cu_seqlens, blockmask, rng_state = ctx.saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        dqkv = _flash_blocksparse_attn_backward(dout, qkv, context, context, softmax_lse, cu_seqlens, blockmask, ctx.dropout_p, ctx.max_s, ctx.softmax_scale, ctx.causal)
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return (dqkv, None, None, None, None, None, None, None)