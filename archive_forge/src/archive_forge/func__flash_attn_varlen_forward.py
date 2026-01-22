from typing import Optional, Union
import torch
import torch.nn as nn
import flash_attn_2_cuda as flash_attn_cuda
def _flash_attn_varlen_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_softmax):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.varlen_fwd(q, k, v, None, cu_seqlens_q, cu_seqlens_k, None, alibi_slopes, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, False, causal, window_size[0], window_size[1], return_softmax, None)
    return (out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state)