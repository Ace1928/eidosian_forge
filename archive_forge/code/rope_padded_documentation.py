from typing import Optional, Tuple
import torch
from xformers.ops.fmha.attn_bias import (  # type: ignore
from .. import _is_triton_available

    Performs RoPE (rotary embeddings) and kv-cache emplacement for a heterogeneous
    batch for inference in the style given by
    BlockDiagonalCausalWithOffsetPaddedKeysMask.
    The batch is concatenated along the sequence dimension, so the
    actual dim-0 length of all tensors is 1.

    xq, xk and xv should be (1, slen, n_heads, dim), where
    xq's n_heads can differ from xk and xv.

    This function places the roped xk in the right place in cache_k, and
    xv (unmodified) in the right place in cache_v, and returns out_q
    (the roped xq) such that things are ready to call

    xformers.ops.memory_efficient_attention(
        out_q, cache_k, cache_v, attn_bias=attn_bias
    )

    This functionality is experimental. Its API might be changed without warnings.
    Use it at your own risk.

    Arguments:
        xq: tensor of queries to apply rope to
        xk: tensor of keys to apply rope to
        xv: tensor of values to copy into cache_v
        cache_k: cache of keys, MODIFIED IN PLACE
        cache_v: cache of values, MODIFIED IN PLACE
        attn_bias: details the layout of caches.
                Used to determine frequencies for the
                RoPE calculation as well as the locations in cache_k and cache_v
                to write to. Must be on the device.
        adjacents: If True, the inputs are in adjacent pairs along the final dim axis.
                  This is like the released LLaMA model.
                  If False, the dim axis is split in two equal pieces.
                   I.e. the features are ordered with all the real parts before all
                   the imaginary parts. This matches HuggingFace, e.g.
                   https://github.com/huggingface/transformers/blob/
                   f143037789288ba532dada934a118e648e715738/
                   src/transformers/models/llama/modeling_llama.py#L126-L130
        internal_dtype: set to "f32" or "f64" to enforce dtype in the calculation
    