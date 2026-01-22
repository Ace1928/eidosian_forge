from typing import Optional, Union
import torch
import torch.nn as nn
import flash_attn_2_cuda as flash_attn_cuda
def _flash_attn_varlen_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic, rng_state=None):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d = flash_attn_cuda.varlen_bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k, alibi_slopes, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, False, causal, window_size[0], window_size[1], deterministic, None, rng_state)
    return (dq, dk, dv, softmax_d)